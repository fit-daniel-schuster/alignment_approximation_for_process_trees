import os
import string
import logging
from datetime import datetime
import random
from typing import List, Dict, Tuple
import numpy
import concurrent.futures
import pickle
from approximate_alignment import approximate_alignment
from calculate_a_sa_ea_sets import initialize_a_sa_ea_tau_sets
from experimental_setup.create_heat_map import create_heat_map
from utilities import get_costs_from_alignment, __calculate_optimal_alignment, trace_to_list_of_str, \
    get_process_tree_height, process_tree_to_binary_process_tree, \
    get_number_leaf_nodes, get_number_nodes, get_number_inner_nodes
from pm4py.objects.log.log import EventLog, Trace, Event
from pm4py.objects.process_tree import semantics
from pm4py.algo.simulation.tree_generator import factory as tree_gen_factory
import pm4py.visualization.process_tree.factory as pt_vis
from pm4py.objects.process_tree.process_tree import ProcessTree
from pm4py.objects.log.importer.xes.factory import import_log
from pm4py.algo.filtering.log.variants import variants_filter


def start_experiments_for_ptml_files(path, file_names_pt, file_name_log, sample_size=None):
    logging.disable(logging.CRITICAL)

    input_data = []
    print("load log")
    log = import_log(path + file_name_log)
    print("finish loading log")
    variants = variants_filter.get_variants(log)
    log_variants = EventLog()
    for v in variants:
        log_variants.append(variants[v][0])
    if sample_size:
        log_variants = random.sample(log_variants, sample_size)
    for ptml_file_name in file_names_pt:
        with open(path + ptml_file_name, "rb") as input_file:
            pt = pickle.load(input_file)
            pt_vis.view(pt_vis.apply(pt, parameters={"format": "svg"}))
            pt = process_tree_to_binary_process_tree(pt)
            pt_vis.view(pt_vis.apply(pt, parameters={"format": "svg"}))
            input_data.append((pt, log_variants))
    start_experiments(input_data=input_data)


def start_experiments(number_process_trees=1, number_traces_per_tree=1,
                      input_data: List[Tuple[ProcessTree, EventLog]] = None):
    logging.disable(logging.CRITICAL)
    results_per_tree = {}

    processed_trees: List[ProcessTree] = []
    futures: Dict[ProcessTree, List[concurrent.futures]] = {}

    if input_data is None:
        input_data = []
        for i in range(number_process_trees):
            parameters = {"min": 20, "mode": 35, "max": 50, "duplicate": 0.25, "silent": .25}
            tree = tree_gen_factory.apply(parameters=parameters)
            tree = process_tree_to_binary_process_tree(tree)
            log = semantics.generate_log(tree, no_traces=number_traces_per_tree)
            log = __introduce_deviations_in_log(log)
            input_data.append((tree, log))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(len(input_data)):
            print("process tree ", i + 1, "/", number_process_trees)
            tree = input_data[i][0]
            log = input_data[i][1]

            results_per_tree[tree] = {}
            processed_trees.append(tree)
            results_per_tree[tree]["log"] = log

            start_time = datetime.now()
            a_sets, sa_sets, ea_sets, tau_sets = initialize_a_sa_ea_tau_sets(tree)
            end_time = datetime.now()
            results_per_tree[tree]["calculation_time_of_a_sa_ea_tau_sets"] = (end_time - start_time).total_seconds()

            results_per_tree[tree]["alignments"] = []
            futures[tree] = []
            for j, trace in enumerate(log):
                futures[tree].append(
                    executor.submit(__calculate_approx_and_opt_align, trace, tree, a_sets, sa_sets, ea_sets,
                                    tau_sets))
        for i, tree in enumerate(processed_trees):
            print("Waiting for process tree ", i + 1, "/", len(processed_trees))
            results_per_tree[tree]["alignments"] = []
            for j, f in enumerate(futures[tree]):
                print("Waiting for trace ", j + 1, "/", len(futures[tree]))
                results_per_tree[tree]["alignments"].append(f.result())
        __save_obj_to_file("results_per_tree_", results_per_tree)


def __calculate_approx_and_opt_align(t: Trace, pt: ProcessTree, s_sets, sa_sets, ea_sets, tau_flag,
                                     calculate_opt_alignments=True):
    logging.disable(logging.CRITICAL)
    # optimal alignment
    start_time_opt = datetime.now()
    if calculate_opt_alignments:
        opt_align = __calculate_optimal_alignment(pt, t)['alignment']
    else:
        opt_align = []
    end_time_opt = datetime.now()
    duration_opt = (end_time_opt - start_time_opt).total_seconds()
    # approximate alignment
    abortion_trace_length = [2, 4, 6, 8]
    abortion_tree_height = [2, 4, 6, 8]
    res = {}
    for tl in abortion_trace_length:
        for th in abortion_tree_height:
            start_time_approx = datetime.now()
            approx_align = approximate_alignment(pt, s_sets, sa_sets, ea_sets, tau_flag, t,
                                                 trace_length_abortion_criteria=tl,
                                                 process_tree_height_abortion_criteria=th)
            end_time_approx = datetime.now()
            duration_approx = (end_time_approx - start_time_approx).total_seconds()
            res[(th, tl)] = {
                "approx_alignment": approx_align,
                "costs_approx_alignment": get_costs_from_alignment(approx_align),
                "duration_approx_alignment": duration_approx,
                "opt_alignment": opt_align,
                "costs_opt_align": get_costs_from_alignment(opt_align),
                "duration_opt_alignment": duration_opt,
                "trace": t
            }
    return res


def __introduce_deviations_in_log(log: EventLog, probability_alter_trace=.5, probability_delete_event=.2,
                                  probability_change_activity_label=.2, probability_insert_events=.2):
    altered_log = EventLog()
    for t in log:
        if random.random() < probability_alter_trace:
            altered_trace = Trace()
            for e in t:
                action = random.choice([1, 2, 3])
                if action == 1 and not random.random() < probability_delete_event:
                    altered_trace.append(e)
                elif action == 2 and random.random() < probability_change_activity_label:
                    e["concept:name"] = random.choice(string.ascii_letters)
                    altered_trace.append(e)
                elif action == 3 and random.random() < probability_insert_events:
                    e = Event()
                    e["concept:name"] = random.choice(string.ascii_letters)
                    altered_trace.append(e)
            altered_log.append(altered_trace)
        else:
            altered_log.append(t)
    return altered_log


def __save_obj_to_file(filename: str, obj):
    with open(filename + "_" + str(datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-") + '.pickle',
              'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def __get_statistics_of_pt(pt: ProcessTree, res=None):
    if res is None:
        res = {"number_leaves": 0,
               "number_visible_leaves": 0,
               "number_tau_leaves": 0,
               "number_inner_nodes": 0,
               "number_nodes": 0,
               "tree_height": get_process_tree_height(pt)}
    res["number_nodes"] += 1
    if pt.children is None or len(pt.children) == 0:
        res["number_leaves"] += 1
        if pt.label is None:
            res["number_tau_leaves"] += 1
        else:
            res["number_visible_leaves"] += 1
        return res
    else:
        res["number_inner_nodes"] += 1
        for c in pt.children:
            res = __get_statistics_of_pt(c, res)
        return res


def __load_pickle_obj(filepath):
    with open(filepath, 'rb') as handle:
        obj = pickle.load(handle)
        return obj


def __analyze_process_tree_characteristics(trees):
    print("---\ntree characteristics\n")
    heights = []
    number_nodes = []
    number_leaf_nodes = []
    number_inner_nodes = []
    for pt in trees:
        number_nodes.append(get_number_nodes(pt))
        heights.append(get_process_tree_height(pt))
        number_leaf_nodes.append(get_number_leaf_nodes(pt))
        number_inner_nodes.append(get_number_inner_nodes(pt))
    print("TH max:", numpy.max(heights))
    print("TH avg:", numpy.average(heights))
    print("TH min:", numpy.min(heights))
    print("TH median:", numpy.median(heights))
    print("avg. leaf nodes:", numpy.average(number_leaf_nodes))
    print("avg. nodes:", numpy.average(number_nodes))
    print("avg. inner nodes:", numpy.average(number_inner_nodes))


def __analyze_trace_characteristics(traces):
    print("---\ntrace characteristics\n")
    lengths = []
    for t in traces:
        lengths.append(len(t))
    print("TL max:", numpy.max(lengths))
    print("TL avg:", numpy.average(lengths))
    print("TL min:", numpy.min(lengths))
    print("TL median:", numpy.median(lengths))


def __analyze_alignments(res_file, save_plots=False):
    logging.disable(logging.CRITICAL)
    res = __load_pickle_obj(res_file)

    calculation_time_a_sa_ea_tau = []

    th = []
    tl = []
    for abortion_criteria in res[list(res.keys())[0]]["alignments"][0]:
        if abortion_criteria[0] not in th:
            th.append(abortion_criteria[0])
        if abortion_criteria[1] not in tl:
            tl.append(abortion_criteria[1])

    approx_calculation_time_per_abortion_criteria = numpy.full((len(th), len(tl)), None)
    approx_costs_per_abortion_criteria = numpy.full((len(th), len(tl)), None)

    duration_opt = []
    costs_opt = []
    traces = []

    for pt in res:
        calculation_time_a_sa_ea_tau.append(res[pt]["calculation_time_of_a_sa_ea_tau_sets"])
        traces.extend(list(res[pt]['log']))
        for a in res[pt]['alignments']:
            costs_opt.append(a[list(a.keys())[0]]['costs_opt_align'])
            duration_opt.append(a[list(a.keys())[0]]['duration_opt_alignment'])

            for criteria in a:
                cost = a[criteria]["costs_approx_alignment"]
                if approx_costs_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])] is None:
                    approx_costs_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])] = []
                approx_costs_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])].append(cost)

                duration = a[criteria]['duration_approx_alignment']
                if approx_calculation_time_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])] is None:
                    approx_calculation_time_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])] = []
                approx_calculation_time_per_abortion_criteria[th.index(criteria[0])][tl.index(criteria[1])].append(
                    duration)

    mean = numpy.vectorize(lambda x: numpy.average(x))
    avg_approx_calculation_time_per_abortion_criteria = mean(approx_calculation_time_per_abortion_criteria)
    avg_approx_costs_per_abortion_criteria = mean(approx_costs_per_abortion_criteria)
    avg_calculation_time_a_sa_ea_tau = numpy.average(calculation_time_a_sa_ea_tau)

    median = numpy.vectorize(lambda x: numpy.median(x))
    median_approx_calculation_time_per_abortion_criteria = median(approx_calculation_time_per_abortion_criteria)
    median_approx_costs_per_abortion_criteria = median(approx_costs_per_abortion_criteria)

    avg_duration_opt = numpy.average(duration_opt)
    avg_costs_opt = numpy.average(costs_opt)

    median_duration_opt = numpy.median(duration_opt)
    median_costs_opt = numpy.median(costs_opt)

    __analyze_trace_characteristics(traces)
    __analyze_process_tree_characteristics(list(res.keys()))

    print("avg. duration opt. alignment: ", avg_duration_opt)
    print("avg. costs opt. alignment: ", avg_costs_opt)
    print("avg. duration calculation A,SA,EA,tau-flag per tree ", avg_calculation_time_a_sa_ea_tau)
    create_heat_map(avg_approx_costs_per_abortion_criteria,
                    title="avg. alignment cost optimal alignments: " + str(round(avg_costs_opt, 2)),
                    save=save_plots, x_labels=tl, y_labels=th)
    # create_surface_plot(avg_approx_costs_per_abortion_criteria)
    create_heat_map(avg_approx_calculation_time_per_abortion_criteria,
                    title="avg. computation time (s) optimal alignments: " + str(round(avg_duration_opt, 2)),
                    save=save_plots, x_labels=tl, y_labels=th)

    # create_heat_map(median_approx_costs_per_abortion_criteria,
    #                 title="median alignment cost optimal alignments: " + str(round(median_costs_opt, 2)),
    #                 save=save_plots, x_labels=tl, y_labels=th)
    # create_heat_map(median_approx_calculation_time_per_abortion_criteria,
    #                 title="median computation time (s) optimal alignments: " + str(round(median_duration_opt, 2)),
    #                 save=save_plots, x_labels=tl, y_labels=th)


if __name__ == '__main__':
    # Experiments
    # start_experiments(number_process_trees=50, number_traces_per_tree=50, input_data=None)
    # dirname = os.path.dirname(__file__)
    # path = dirname + '/bpi_ch_19/'
    # start_experiments_for_ptml_files(path, ["tree_imf_.7.ptml.pickle"], "log.xes", sample_size=100)
    # path = dirname + '/bpi_ch_18/'
    # start_experiments_for_ptml_files(path, ["tree_imf_.7.ptml.pickle"], "log.xes", sample_size=100)

    # Results
    __analyze_alignments("results_per_tree__2020-07-14-21-23-59_50_trees_without_duplicates_50_traces.pickle")
    __analyze_alignments("results_per_tree__2020-07-14-22-00-16_50_trees_with_duplicates_50_traces.pickle")
    __analyze_alignments("results_per_tree__2020-07-15-14-03-39-543079_bpi_ch_19_100_variants.pickle")
    __analyze_alignments("results_per_tree__2020-07-15-18-52-44-379162_bpi_ch_18_100_variants.pickle")
