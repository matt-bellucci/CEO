from __future__ import annotations
from abc import ABC, abstractmethod

import copy
from typing import Union

import owlready2 as owl

import onto_utils

Primitives = Union[int, bool, float, str]


class Assertion(ABC):

    def __init__(self, property, instance):
        self.property = property
        self.instance = instance

    @abstractmethod
    def get_property_name(self):
        pass

    @abstractmethod
    def get_instance_name(self):
        pass

    @abstractmethod
    def get_instance_type(self):
        pass

    def __str__(self):
        return f"({self.get_property_name()} {self.get_instance_name()} is a {self.get_instance_type()})"

    def __hash__(self):
        return hash((self.property, self.instance))

    def __eq__(self, other):
        return other is not None and self.property == other.property and self.instance == other.instance


class ObjectAssertion(Assertion):
    def __init__(self, property: owl.ObjectPropertyClass, instance: owl.entity):
        super().__init__(property, instance)

    def get_property_name(self):
        return self.property.name

    def get_instance_name(self):
        return self.instance.name

    def get_instance_type(self):
        return self.instance.is_a

    def __deepcopy__(self, memodict={}):
        return ObjectAssertion(self.property, self.instance)


class DataAssertion(Assertion):

    def __init__(self, property: owl.DataPropertyClass, instance: Primitives):
        super().__init__(property, instance)
        self.datatypes = property.range

    def get_property_name(self):
        return self.property.name

    def get_instance_name(self):
        return str(self.instance)

    def get_instance_type(self):
        return type(self.instance)

    def __deepcopy__(self, memodict={}):
        return DataAssertion(self.property, self.instance)


def extract_assertions_from_owl_individual(owl_individual):
    properties = owl_individual.get_properties()
    assertions = []
    for property in properties:
        if isinstance(property, owl.DataPropertyClass):
            assertion_type = DataAssertion
        else:
            assertion_type = ObjectAssertion
        for value in property[owl_individual]:
            assertion = assertion_type(property, value)
            assertions.append(assertion)
    return assertions


def create_individual_from_ontology(owl_individual):
    assertions = extract_assertions_from_owl_individual(owl_individual)
    is_a = owl_individual.is_a
    return Individual(assertions, is_a)


def choose_assertion_type(property):
    if isinstance(property, owl.ObjectPropertyClass):
        return ObjectAssertion
    elif isinstance(property, owl.DataPropertyClass):
        return DataAssertion
    else:
        return None


def sorted_assertions(assertions):
    return sorted(assertions, key=lambda x: str(x))


class Individual:
    def __init__(self, assertions: list[Assertion], is_a, is_consistent=None):
        self.assertions = sorted_assertions(assertions)
        self.is_a = is_a
        self.is_consistent = is_consistent
        self.assertions_str_dict = {
            (assertion.get_property_name(), assertion.get_instance_name()): assertion for assertion in self.assertions
        }

    def __str__(self):
        return f"Assertions: {[str(assertion) for assertion in sorted_assertions(self.assertions)]}"

    def __hash__(self):
        return hash((tuple(sorted_assertions(self.assertions)), tuple(self.is_a), len(self.assertions)))

    def __eq__(self, other):
        return other is not None and set(self.assertions) == set(other.assertions) \
               and set(self.is_a) == set(other.is_a) \
               and len(self.assertions) == len(other.assertions)

    def change_assertion_instance(self, assertion, new_instance):
        assertion_to_modify_index = self.assertions.index(assertion)
        self.assertions[assertion_to_modify_index].instance = new_instance

    def add_assertion(self, assertion: Assertion):
        self.assertions.append(assertion)
        self.is_consistent = None

    def set_classes(self, is_a):
        self.is_a = is_a
        self.is_consistent = None

    def check_consistency(self, ontology, name="consistencyCheck", destroy=True, return_inconsistent=False):
        if self.is_consistent is not None:
            return self.is_consistent
        default_class = list(ontology.classes())[0]
        new_individual = default_class(name)
        new_individual.is_a = self.is_a
        for assertion in self.assertions:
            onto_utils.add_relation_to_indiv(new_individual, assertion.property, assertion.instance)
        if return_inconsistent:
            consistent, explanations = onto_utils.is_consistent(ontology, return_explanations=return_inconsistent)
            inconsistent_assertions = self._extract_inconsistent_assertion(explanations, name)
        else:
            consistent = onto_utils.is_consistent(ontology)
        if destroy:
            owl.destroy_entity(new_individual)
        self.is_consistent = consistent
        if return_inconsistent:
            return consistent, inconsistent_assertions
        else:
            return consistent

    def __deepcopy__(self, memodict={}):
        return Individual(copy.deepcopy(self.assertions, memodict), copy.deepcopy(self.is_a, memodict),
                          None)

    def _extract_inconsistent_assertion(self, explanation, name):
        explanations = explanation.splitlines()
        explanations = [expl.split() for expl in explanations]
        assertions_properties_dict = {assertion.property.name: assertion.property for assertion in self.assertions}
        indiv_assertions = [self.assertions_str_dict[(expl[1], expl[2])] for expl in explanations
                            if name in expl and expl[1] in assertions_properties_dict.keys()]

        return indiv_assertions


class Operation(ABC):
    pass


class AssertionRemovalOperation(Operation):
    def __init__(self, removed_assertion):
        # Assertion is a tuple (property, instance)
        self.removed_assertion = removed_assertion

    def __str__(self):
        return f"Removal Operation: {self.removed_assertion}"

    def __eq__(self, other):
        return other is not None and self.removed_assertion == other.removed_assertion

    def __hash__(self):
        return hash(("Remove", self.removed_assertion))


class AssertionInsertionOperation(Operation):
    # Assertion is a tuple (property, instance)
    def __init__(self, added_assertion):
        self.added_assertion = added_assertion

    def __eq__(self, other):
        return other is not None and self.added_assertion == other.added_assertion

    def __hash__(self):
        return hash(("Add", self.added_assertion))


class ClassModificationOperation(Operation):
    def __init__(self, previous_instance, new_instance, property):
        self.previous_instance = previous_instance
        self.new_instance = new_instance
        self.property = property

    def __eq__(self, other):
        return other is not None \
               and self.property == other.property \
               and self.previous_instance == other.previous_instance \
               and self.new_instance == other.new_instance

    def __hash__(self):
        return hash((self.property, self.previous_instance, self.new_instance))
