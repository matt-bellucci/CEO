import copy
from itertools import product, combinations
from typing import Union, Callable

import owlready2 as owl
import networkx as nx
import matplotlib.pyplot as plt
import onto_utils as utils
from graph import Individual, create_individual_from_ontology, AssertionRemovalOperation, \
    AssertionInsertionOperation, ClassModificationOperation, Operation, choose_assertion_type, Assertion, \
    ObjectAssertion


def check_list_intersection(list_1: list, list_2: list) -> bool:
    """
    Checks if two lists have at least one element in common.

    :param list_1: Input list
    :param list_2: Input list
    :return: True when the two lists share an element, false otherwise
    """
    return not set(list_1).isdisjoint(list_2)


def compute_neighbors_remove_assertion(graph: nx.DiGraph, source_individual: Individual,
                                       non_actionnable_property: owl.ObjectProperty = None) -> [nx.DiGraph,
                                                                                                list[Individual]]:
    """
    Computes every neighbor of source_individual, connected by removing an assertion from source_individual.

    :param graph: The initial oriented graph
    :param source_individual: The node to compute the neighbors of
    :param non_actionnable_property: If a property is a subproperty of non_actionnable_property,
                                    then modifying assertions with such property will not be explored.
    :return: The initial graph with the added neighbors and a list of the added neighbors
    """
    source_assertions = source_individual.assertions
    individuals = []
    # Iteratively remove an assertion from the source_individual to create a neighbor
    for assertion_to_remove in source_assertions:
        # If property is not actionnable, skip it.
        if utils.is_subproperty_of(assertion_to_remove.property, non_actionnable_property):
            continue

        individual = remove_assertion_from_indiv(source_individual, assertion_to_remove)

        # add the individual to the graph
        graph.add_node(individual)
        graph.add_edge(source_individual, individual, operation=AssertionRemovalOperation(assertion_to_remove))

        individuals.append(individual)
    return graph, individuals


def remove_assertion_from_indiv(source_individual: Individual, assertion_to_remove: Assertion) -> Individual:
    """
    Removes a given assertion from an individual.

    :param source_individual: The individual to remove an assertion from.
    :param assertion_to_remove: The assertion to remove
    :return: A new individual similar to source_individual with the assertion removed
    """
    new_assertions = [asser for asser in source_individual.assertions if asser != assertion_to_remove]
    is_a = source_individual.is_a
    return Individual(new_assertions, is_a)


def generate_one_ofs(graph, source_individual, assertion, one_ofs, ontology):
    any_consistent = False
    for one_of in one_ofs:
        if assertion.instance not in one_of.instances:
            continue
        for instance in one_of.instances:
            if instance == assertion.instance:
                continue
            new_indiv = copy.deepcopy(source_individual)
            new_indiv.change_assertion_instance(assertion, instance)
            is_consistent = new_indiv.check_consistency(ontology)
            graph.add_node(new_indiv)
            graph.add_edge(source_individual, new_indiv,
                           operation=ClassModificationOperation(assertion.instance, instance, assertion.property))
            if is_consistent:
                any_consistent = True
    return graph, any_consistent


def explore_and_generate(graph: Union[None, nx.DiGraph], ontology: owl.Ontology,
                         source_individual: Individual, wanted_is_a: list[owl.ThingClass],
                         non_actionnable_property: owl.ObjectProperty = None,
                         use_naive: bool = False) -> nx.DiGraph:
    """
    Searches consistent individuals in an unknown graph of individuals, starting from source_individual.
    This function explores neighbors of source_individual by removing assertions until a consistent individual is found.
    It uses the output of the reasoner to know which assertions cause the inconsistency.


    :param graph: The oriented graph to explore and generate
    :param ontology: The ontology used to check the consistency of the individuals.
    :param source_individual: The individual from which to start exploring.
    :param wanted_is_a: The desired class of source_individual.
    :param non_actionnable_property: If a property is a subproperty of non_actionnable_property,
                                    then modifying assertions with such property will not be explored.
    :param use_naive:  Whether to use the naive exploration algorithm.
    :return: The graph with the new individuals.
    """
    if graph is None:
        graph = nx.DiGraph()
    source_individual.set_classes(wanted_is_a)
    _, inconsistent_assertions = source_individual.check_consistency(ontology, return_inconsistent=True)
    graph.add_node(source_individual)

    if source_individual.is_consistent:
        return graph

    if use_naive or not inconsistent_assertions:
        # if inconsistent but no assertions isolated, then naive exploration
        return explore_and_generate_naive(graph, ontology, source_individual, wanted_is_a,
                                          non_actionnable_property=non_actionnable_property)

    else:
        # When an inconsistency is detected, remove the responsible assertion(s)
        for assertion in inconsistent_assertions:
            # If property is not actionnable, skip it.
            if utils.is_subproperty_of(assertion.property, non_actionnable_property):
                continue
            # If the assertion to remove has an instance of a class that is defined by a OneOf, check the other
            # instances of the OneOf for any consistency. If a consistent instance found, stop here.
            if isinstance(assertion, ObjectAssertion):
                one_ofs = utils.has_one_of(assertion.get_instance_type())
                graph, any_consistent = generate_one_ofs(graph, source_individual, assertion, one_ofs, ontology)
                if any_consistent:
                    return graph

            individual = remove_assertion_from_indiv(source_individual, assertion)
            if not graph.has_node(individual):

                # Recursively repeat this process until the individual is consistent.
                consistent, inconsistent_assertions = individual.check_consistency(ontology, return_inconsistent=True)
                graph.add_node(individual)
                graph.add_edge(source_individual, individual, operation=AssertionRemovalOperation(assertion))
                if not consistent:
                    # If the new individual is not consistent,
                    # we keep exploring by removing the new problematic assertion,
                    # then we go back to the original individual and remove the new problematic assertion.
                    other_indiv = remove_assertion_from_indiv(source_individual, inconsistent_assertions[0])
                    if not graph.has_node(other_indiv):
                        other_graph = explore_and_generate(graph, ontology, other_indiv, wanted_is_a,
                                                           non_actionnable_property=non_actionnable_property)
                        graph = nx.compose(graph, other_graph)
                other_graph = explore_and_generate(graph, ontology, individual, wanted_is_a,
                                                   non_actionnable_property=non_actionnable_property)
                graph = nx.compose(graph, other_graph)

    return graph


def explore_and_generate_naive(graph: Union[None, nx.DiGraph], ontology: owl.Ontology,
                               source_individual: Individual, wanted_is_a: list[owl.ThingClass],
                               non_actionnable_property: owl.ObjectProperty = None) -> nx.DiGraph:
    """
    Searches consistent individuals in an unknown graph of individuals, starting from source_individual.
    This function explores neighbors of source_individual by removing assertions until a consistent individual is found.
    It explores by naively removing assertions one by one,
    until no assertions are left or a consistent individual is found.

    :param graph: The oriented graph to explore and generate
    :param ontology: The ontology used to check the consistency of the individuals.
    :param source_individual: The individual from which to start exploring.
    :param wanted_is_a: The desired class of source_individual.
    :param non_actionnable_property: If a property is a subproperty of non_actionnable_property,
                                    then modifying assertions with such property will not be explored.
    :return: The graph with the new individuals.
    """
    if graph is None:
        graph = nx.DiGraph()
    source_individual.set_classes(wanted_is_a)
    source_individual.check_consistency(ontology)
    graph.add_node(source_individual)

    if source_individual.is_consistent:
        return graph
    else:
        graph, neighbors = compute_neighbors_remove_assertion(graph, source_individual,
                                                              non_actionnable_property=non_actionnable_property)
        for neighbor in neighbors:
            other_graph = explore_and_generate_naive(None, ontology, neighbor, wanted_is_a,
                                                     non_actionnable_property=non_actionnable_property)
            graph = nx.compose(graph, other_graph)
    return graph


def generate_ancestors(graph: nx.DiGraph, ontology: owl.Ontology, individual: Individual) -> nx.DiGraph:
    """
    Finds the individual's predecessors in the graph, that are connected via assertion removal.
    The function adds the removed assertion back and generates new individuals by modifying the removed assertion's
    instance class to an ancestor class. It stops when a consistent individual is found
    or when the property range is reached.

    :param graph: The oriented graph containing individuals.
    :param ontology: The ontology used to check the consistency of the individuals.
    :param individual: The individual from which to start exploring. This individual should be consistent in order to
    generate counterfactuals explanations.
    :return: A graph with the individual's predecessors.
    """
    individual_predecessors = graph.predecessors(individual)
    for predecessor in individual_predecessors:
        operation = graph.edges[predecessor, individual]["operation"]
        if not isinstance(operation, AssertionRemovalOperation):
            continue

        # Searching for consistent ancestors of the removed assertion's instance
        removed_assertion = operation.removed_assertion
        property_range = removed_assertion.property.range
        ancestors = removed_assertion.instance.is_a
        assertion_type = choose_assertion_type(removed_assertion.property)
        consistent = False

        # If the assertion to remove has an instance of a class that is defined by a OneOf, check the other
        # instances of the OneOf for any consistency.
        if assertion_type == ObjectAssertion:
            one_ofs = utils.has_one_of(removed_assertion.get_instance_type())
            graph, any_consistent = generate_one_ofs(graph, predecessor, removed_assertion, one_ofs, ontology)
            consistent = any_consistent

        while not consistent and not check_list_intersection(property_range, ancestors):
            # while not consistent and the property range is not reached.
            ancestors = utils.get_class_parents(ancestors)
            # Modify the instance's class with one of the ancestor and add the new individual in the graph.
            for ancestor in ancestors:
                ancestor_instance = utils.get_class_individual(ancestor)
                ancestor_individual = copy.deepcopy(individual)
                new_assertion = assertion_type(removed_assertion.property, ancestor_instance)
                ancestor_individual.add_assertion(new_assertion)
                is_ancestor_consistent = ancestor_individual.check_consistency(ontology)
                graph.add_node(ancestor_individual)
                operation = AssertionInsertionOperation(new_assertion)
                graph.add_edge(individual, ancestor_individual, operation=operation)
                if is_ancestor_consistent:
                    # if one is True in for loop, it must remain True for the rest of the loop
                    consistent = True
    return graph


def generate_all_ancestors(graph: nx.DiGraph, ontology: owl.Ontology, max_iterations=None) -> nx.DiGraph:
    """
    Generate the ancestors of every node in the graph.

    :param graph: The oriented graph containing individuals.
    :param ontology: The ontology used to check the consistency of the individuals.
    :param max_iterations: Max number of ancestor generation. If None, stops when converged.
    :return: A graph with every ancestor of every node.
    """
    # graph.nodes is modified when adding new nodes so we store the preexisting nodes
    nodes = list(graph.nodes)
    n_nodes = 0
    i = 0
    while len(nodes) > n_nodes:
        for node in nodes:
            graph = generate_ancestors(graph, ontology, node)
        graph = connect_all_nodes(graph)
        n_nodes = len(nodes)
        nodes = list(graph.nodes)
        i += 1
        if max_iterations and i >= max_iterations:
            break
    return graph


def generate_individual_descendants(graph: nx.DiGraph, ontology: owl.Ontology,
                                    individual: Individual, only_consistent: bool = True):
    """
    Modifies each assertion by modifying the class of its instance to a direct descendant.
    For every assertion, a new individual is created and added to the graph, connected directly to the main individual.
    This function is used with a consistent individual to find more consistent individuals.

    :param graph: The oriented graph containing individuals.
    :param ontology: The ontology used to check the consistency of the individuals.
    :param individual: The individual from which to start exploring. This individual should be consistent in order to
    generate counterfactuals explanations.
    :param only_consistent: Whether to stop the exploration when an inconsistent individual is created.
    :return: A larger graph with the individual's descendants.
    """
    assertions = individual.assertions
    for assertion in assertions:
        instance_class = assertion.instance.is_a
        for cls in instance_class:
            direct_descendants = utils.get_class_descendants(ontology, cls)
            for descendant in direct_descendants:
                # Create a new individual with the modified assertion, by copying the current one and modifying the
                # desired assertion with a new instance of the wanted class.
                descendant_instance = utils.get_class_individual(descendant)
                descendant_indiv = copy.deepcopy(individual)
                descendant_indiv.change_assertion_instance(assertion, descendant_instance)
                # assertion_to_modify_index = descendant_indiv.assertions.index(assertion)
                # descendant_indiv.assertions[assertion_to_modify_index].instance = descendant_instance
                if descendant_indiv not in graph.nodes:
                    descendant_indiv.check_consistency(ontology)
                    graph.add_node(descendant_indiv)
                    operation = ClassModificationOperation(assertion.instance, descendant_instance, assertion.property)
                    graph.add_edge(individual, descendant_indiv, operation=operation)
                if not only_consistent or descendant_indiv.is_consistent:
                    graph = generate_individual_descendants(graph, ontology, descendant_indiv)
    return graph


def generate_all_individual_descendants(graph: nx.DiGraph, ontology: owl.Ontology,
                                        only_consistent: bool = True) -> nx.DiGraph:
    """
    Generate the descendants of every node in the graph.

    :param graph: The oriented graph containing individuals.
    :param ontology: The ontology used to check the consistency of the individuals.
    :param only_consistent: Whether to stop exploring when an inconsistent individual is created.
    :return: A graph with every descendant of every node.
    """
    # graph.nodes is modified when adding new nodes so we store the preexisting nodes
    nodes = list(graph.nodes)
    for node in nodes:
        graph = generate_individual_descendants(graph, ontology, node, only_consistent=only_consistent)
    return graph


def connect_all_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Iterates through all nodes of the graph to connect them when possible.

    :param graph: The oriented graph containing individuals.
    :return: The oriented graph with new edges.
    """
    for node_source in graph:
        # Iterate through all nodes
        for node_target in graph:
            # Iterate through all nodes that are not connected to node_source
            if node_source == node_target:
                continue
            if graph.has_edge(node_source, node_target):
                continue
            # Get the type of connection between the two nodes
            operation = get_node_link(node_source, node_target)
            if operation is not None:
                graph.add_edge(node_source, node_target, operation=operation)
    return graph


def get_node_link(node_source: Individual, node_target: Individual) -> Union[None, Operation]:
    """
    Identifies the type of connection between two nodes of a graph.
    Only nodes of the same class and that needs a single operation (assertion removal or insertion or class modification
    of one assertion).

    :param node_source: The node at the start of the edge.
    :param node_target: The node at the end of the edge.
    :return: The Operation that connects both nodes or None if the nodes cannot be connected.
    """

    # Only nodes of the same class can be connected
    if node_source.is_a != node_target.is_a:
        return None
    # Get the dictionaries of assertions, of the form
    # {property1: [instance1, instance2],
    #  property2: [instance3]}
    source_assertions_dict = assertions_list_to_dict(node_source.assertions)
    target_assertions_dict = assertions_list_to_dict(node_target.assertions)

    # Get the properties that are not shared by the two individuals
    properties_difference = set(source_assertions_dict.keys()) ^ set(target_assertions_dict.keys())
    if len(properties_difference) > 1:
        # If more than one property is not shared, then the nodes are not connected
        return None

    # Get the shared properties
    properties_intersection = set(source_assertions_dict.keys()) & set(target_assertions_dict.keys())
    different_properties = []
    for property in properties_intersection:
        # Find the differences in the instances of every shared property.
        if sorted(source_assertions_dict[property], key=lambda x: x.name) != sorted(target_assertions_dict[property], key=lambda x: x.name):
            different_properties.append(property)
        if len(different_properties) + len(properties_difference) > 1:
            # If more than one property has different instances, or if one property isn't shared and a shared property
            # doesn't have the same instances,
            # then more than one operation is required and the two nodes are not connected
            return None

    if len(different_properties) == 1:
        # If both nodes share every property but list of instances is different for only one property
        # Get the list of instances of the property for both nodes and find the differences
        source_prop_instances = sorted(source_assertions_dict[different_properties[0]], key=lambda x: x.name)
        target_prop_instances = sorted(target_assertions_dict[different_properties[0]], key=lambda x: x.name)
        instance_differences = []
        for x, y in zip(source_prop_instances, target_prop_instances):
            if x != y:
                instance_differences += [x, y]
        # instance_differences = set(source_prop_instances) ^ set(target_prop_instances)
        if len(source_prop_instances) != len(target_prop_instances):
            # If there is not the same number of instances between two nodes, at least one operation is required
            if len(source_prop_instances) > len(target_prop_instances):
                instance_differences += source_prop_instances[len(target_prop_instances):]
            else:
                instance_differences += target_prop_instances[len(source_prop_instances):]
            if len(instance_differences) > 1:
                # If there is more than one different instances, then more than two operations required, not connected
                return None
            else:
                # Only one different instance, meaning that an instance must be added or removed.
                for instance_difference in instance_differences:  # Fastest way to access an element of a set
                    break
                # Find the instance to add or remove
                source_target_instance_difference = set(source_prop_instances) - set(target_prop_instances)
                assertion_type = choose_assertion_type(different_properties[0])
                assertion_to_change = assertion_type(different_properties[0], instance_difference)
                if source_target_instance_difference:
                    return AssertionRemovalOperation(assertion_to_change)
                else:
                    return AssertionInsertionOperation(assertion_to_change)

        else:
            # If the number of instances is the same.
            if len(instance_differences) != 2:
                # If only one instance is different, the operator ^ will return 2 differences.
                # Example: a = set([1, 2]), b = set([1, 3]), a ^ b = {2, 3}.
                # If only one instance is different, then the length of instance differences is 2,
                # Less than 2 means the number of instances is different and this case is already handled
                # more than 2 means that there more than one difference in instances, which would require more than one
                # operation, so not connected.
                return None
            source_instance = [x for x in instance_differences if x in source_prop_instances][0]
            target_instance = [x for x in instance_differences if x in target_prop_instances][0]
            return ClassModificationOperation(source_instance, target_instance, different_properties[0])

    elif len(properties_difference) == 1:
        # If one property must be added or removed
        for prop in properties_difference:  # Fastest way to access an element in a set
            break
        assertion_type = choose_assertion_type(prop)
        prop_in_source = prop in source_assertions_dict  # Check if property is already in source
        if prop_in_source and len(source_assertions_dict[prop]) == 1:
            # If property is in source and has one instance, it must be removed from source
            return AssertionRemovalOperation(assertion_type(prop, source_assertions_dict[prop][0]))
        elif not prop_in_source and len(target_assertions_dict[prop]) == 1:
            # If property is in target and has one instance, it must be added to source
            return AssertionInsertionOperation(assertion_type(prop, target_assertions_dict[prop][0]))
        else:
            # If there is more than one instance to add, then more than two operations must be done, therefore nodes
            # not connected
            return None

    else:
        return None


def assertions_list_to_dict(assertions: list[Assertion]) -> dict[owl.ObjectPropertyClass, owl.entity]:
    """
    Transforms a list of assertions into a dictionary of the form {property: [instance1, instance2, ...]}.

    Example:
    > assertions = [Assertion(prop1, instance1), Assertion(prop1, instance2),
    > Assertion(prop2, instance3), Assertion(prop3, instance4)]
    > assertions_list_to_dict(assertions)
    {prop1: [instance1, instance2],
     prop2: [instance3],
     prop3: [instance4]
    }

    :param assertions: A list of assertions
    :return: A dictionary of the form {property: instances}
    """
    assertions_dict = {}
    for assertion in assertions:
        prop = assertion.property
        instance = assertion.instance
        if prop in assertions_dict:
            assertions_dict[prop].append(instance)
        else:
            assertions_dict[prop] = [instance]
    return assertions_dict


def create_compute_distance_function(ontology: owl.Ontology) -> Callable:
    """
    Creates a distance function that can be used with networkX methods, i.e. that has the form f(source, target, *args)

    :param ontology: The ontology used to check the consistency of the individuals.
    :return: A function to compute the distance between two nodes using their operation attribute.
    """
    _, ontology_shortest_lengths = ontology2graph(ontology)
    ontology_shortest_lengths = dict(ontology_shortest_lengths)

    def compute_distance(source, target, attributes):
        operation = attributes["operation"]
        return compute_operation_distance(operation, ontology_shortest_lengths)

    return compute_distance


def compute_operation_distance(operation: Operation, graphed_ontology_distances):
    """
    Helper function to compute the distance between two nodes based on the operation that connects them.
    A distance per type of operation is defined.

    :param operation: An assertion operation that connects two nodes together.
    :param graphed_ontology_distances:
    :return:
    """
    # use all_pairs_dijkstra to compute all shortest paths between every node, then access simply access it.
    if isinstance(operation, AssertionRemovalOperation):
        return compute_assertion_removal_distance(operation, graphed_ontology_distances)
    elif isinstance(operation, AssertionInsertionOperation):
        return compute_assertion_insertion_distance(operation, graphed_ontology_distances)
    elif isinstance(operation, ClassModificationOperation):
        return compute_class_modification_distance(operation, graphed_ontology_distances)
    else:
        raise TypeError("Input is not the right type")


def compute_assertion_removal_distance(operation, graphed_ontology_distances):
    # Distance for removing an assertion is greater than putting the instance to the most global class
    # Distance for removing an assertion is greater than the maximum path for modifying the same assertion
    assertion = operation.removed_assertion
    property_range = assertion.property.range
    descendants = utils.get_all_descendants(property_range)
    possible_pairs = list(combinations(descendants, 2))
    pair_distances = [graphed_ontology_distances[src][target] for src, target in possible_pairs]
    if not pair_distances:
        return 2  # minimum cost of modification is 1, therefore minimum cost of removal is 2
    else:
        return max(pair_distances) + 2  # must be higher than modification,
        # highest cost of modification is max(pair_distance) + 1 to which we add the cost of removal (+1) thus + 2


def compute_assertion_insertion_distance(operation, graphed_ontology_distances):
    # Distance for inserting an assertion is greater than removing the same assertion, it is penalized
    assertion = operation.added_assertion
    temp_removal_assertion = AssertionRemovalOperation(assertion)
    removal_distance = compute_assertion_removal_distance(temp_removal_assertion, graphed_ontology_distances)
    return removal_distance + 1


def compute_class_modification_distance(operation, graphed_ontology_distances, aggregate=max):
    source_classes = operation.previous_instance.is_a
    target_classes = operation.new_instance.is_a
    return compute_distance_multiple_classes(source_classes, target_classes, graphed_ontology_distances,
                                             aggregate=aggregate) + 1
    # any modification is farther away than no change.no change means distance of 0,
    # when the class is the same but the instance is different, the function compute_distance_multiple_classes returns 0
    # therefore we add 1 to take this case into account.


def compute_distance_multiple_classes(source_classes, target_classes, graphed_ontology_distances, aggregate=max):
    source_target_combinations = product(source_classes, target_classes)
    shortest_lengths = [graphed_ontology_distances[src][target] for src, target in source_target_combinations]
    return aggregate(shortest_lengths)


def ontology2graph(ontology):
    classes = ontology.classes()
    graph = nx.Graph()
    graph.add_nodes_from(classes)
    graph.add_node(owl.Thing)
    for node in graph.nodes:
        descendants = utils.get_class_descendants(ontology, node)
        ancestors = utils.get_class_parents(node)
        linked_classes = descendants.union(ancestors)
        edges = [(node, cls) for cls in linked_classes]
        graph.add_edges_from(edges)
    ontology_shortest_lengths = nx.all_pairs_shortest_path_length(graph)
    return graph, ontology_shortest_lengths


def get_modification_list(node_source, node_target):
    source_assertions = node_source.assertions
    target_assertions = node_target.assertions
    untouched_assertions = set(source_assertions) & set(target_assertions)
    source_assertions_modified = [assertion for assertion in source_assertions if assertion not in untouched_assertions]
    target_assertions_modified = [assertion for assertion in target_assertions if assertion not in untouched_assertions]
    done_target_assertions = []
    class_modifications = []
    for src_assertion in source_assertions_modified:
        for target_assertion in target_assertions_modified:
            if target_assertion not in done_target_assertions:
                if src_assertion.property == target_assertion.property:
                    class_modifications.append((src_assertion, target_assertion))
                    done_target_assertions.append(target_assertion)
    added_assertions = [assertion for assertion in target_assertions_modified if
                        assertion not in done_target_assertions]
    removed_assertions = [assertion for assertion in source_assertions_modified if assertion not in
                          [a[0] for a in class_modifications]
                          ]
    output = {"unmodified": untouched_assertions,
              "modified": class_modifications,
              "added": added_assertions,
              "removed": removed_assertions
              }
    return output


def generate_counterfactuals(ontology, ontology_individual, wanted_class, display_graph=False,
                             non_actionnable_property: owl.ObjectProperty = None, use_naive=True):
    print("create_indiv")
    indiv = create_individual_from_ontology(ontology_individual)
    owl.destroy_entity(ontology_individual)
    print("explore and generate")
    graph = explore_and_generate(None, ontology, indiv, wanted_class, non_actionnable_property=non_actionnable_property,
                                 use_naive=use_naive)
    print("generate ancestors")
    graph = generate_all_ancestors(graph, ontology, max_iterations=2)
    print("generate individuals")
    graph = generate_all_individual_descendants(graph, ontology)
    graph = connect_all_nodes(graph)
    distance_func = create_compute_distance_function(ontology)
    shortest_paths = nx.single_source_dijkstra(graph, indiv, weight=distance_func)
    counterfactuals = {}
    for target in shortest_paths[0].keys():
        if target.is_consistent:
            counterfactuals[target] = {"distance": shortest_paths[0][target],
                                       "modifications": get_modification_list(indiv, target)
                                       }
    if display_graph:
        edges_color = []
        operations = nx.get_edge_attributes(graph, "operation").values()
        for operation in operations:
            if isinstance(operation, AssertionRemovalOperation):
                edges_color.append("red")
            elif isinstance(operation, AssertionInsertionOperation):
                edges_color.append("green")
            elif isinstance(operation, ClassModificationOperation):
                edges_color.append("blue")
            else:
                edges_color.append("blue")
        nodes_color = []
        node_labels = {node: str(node) for node in graph.nodes}
        for node in graph.nodes:
            if node == indiv and not node.is_consistent:
                nodes_color.append("orange")
            elif node.is_consistent is None:
                nodes_color.append("blue")
            elif node.is_consistent:
                nodes_color.append("green")
            elif not node.is_consistent:
                nodes_color.append("red")
            else:
                nodes_color.append("black")
        nx.draw(graph, edge_color=edges_color, node_color=nodes_color, labels=node_labels)
        plt.show()
    return counterfactuals


def test_counterfactuals(ontology, individual, wanted_class, display_graph=True,
                         non_actionnable_property: owl.ObjectProperty = None, use_naive=True):
    counterfactuals = generate_counterfactuals(ontology, individual, wanted_class, display_graph=display_graph,
                                               non_actionnable_property=non_actionnable_property, use_naive=use_naive)
    i = 1
    for indiv, info in counterfactuals.items():
        print(i)
        print(indiv)
        print(f"Distance = {info['distance']}")
        print()
        i += 1
