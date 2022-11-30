import owlready2 as owl
from typing import Union
from collections.abc import Iterable
from custom_reasoning import sync_reasoner_pellet


def load_ontology(path: str) -> owl.namespace.Ontology:
    """
    Loads ontology from a file.
    :param path: The path to the ontology file.
    :return: The ontology
    """
    return owl.get_ontology("file://" + path).load()


def check_relation(restriction: Union[owl.entity.ThingClass, owl.class_construct.ClassConstruct],
                   property: owl.prop.ObjectPropertyClass,
                   subject: owl.entity.ThingClass) -> bool:
    """
    Checks if a given relation corresponds to a restriction of type ObjectProperty.some(object).
    This function is used to find if a given class is a subclass of
    the specific restriction ObjectProperty.restriction(object) where restriction can be any restriction.
    The function also checks the relations of ancestors.

    :param restriction: The restriction to check, usually an element of Class.is_a or Class.equivalent_to.
    :param property: The object property to find in the relation.
    :param subject: The class subject of the relation.
    :return: Whether the relation can be found in the given restriction.
    """

    # if restriction is not an instance of ClassConstruct, then it contains no object property, it usually corresponds
    # to a class.
    if isinstance(restriction, owl.class_construct.ClassConstruct):
        # if it is an instance of Restriction, then it corresponds to what we are looking for.
        # if not, restriction may be of the form And([restriction1, restriction2]). If it is Not() then the relation is
        # false. For other types, such as And() and Or(), check the inner restrictions.
        if isinstance(restriction, owl.class_construct.Restriction):
            return restriction.__dict__["property"] == property and restriction.__dict__["value"] == subject
        elif isinstance(restriction, owl.class_construct.LogicalClassConstruct):
            # if of the form [restriction1 & restriction2 &...],
            # inner_restriction are obtained with .Classes of the main restriction.
            # Check for the relation recursively inside these inner restrictions.
            return any(
                [check_relation(inner_restriction, property, subject) for inner_restriction in restriction.Classes])
        else:
            return False
    else:
        return False


def get_classes_triplet(ontology: owl.namespace.Ontology,
                        property: owl.prop.ObjectPropertyClass,
                        object_class: owl.entity.ThingClass,
                        only_child: bool = True,
                        ignore_classes=None) -> list[owl.entity.ThingClass]:
    """
    Gets all classes that have a restriction corresponding to property.Restriction(object).
    For example, this function will return every class that is defined by a
    relation such as hasProperty.some(Class). The definition includes the definitions of parent classes.
    If one ancestor of a class has the relation that is searched, this class will be returned.
    When the argument only_child is true, it returns only the descendants, otherwise it will return the ancestors
    as well.

    :param ontology: The ontology, in order to access every class.
    :param property: The object property to find in the relation.
    :param object_class: The class subject of the relation
    :param only_child: If True, returns only the bottom-level classes i.e. the classes that have no descendants.
            Otherwise, returns every class and its descendants.
    :param ignore_classes: Classes to be ignored when getting class properties.
    :return: A list of all the classes that contain the given relation.
    """
    if ignore_classes is None:
        ignore_classes = []
    classes = ontology.classes()
    object_descendants = object_class.descendants()
    has_prop = []
    for cls in classes:
        # if range_class ancestor in ignore_classes, then skip class
        if [i for i in list(cls.ancestors()) if i in ignore_classes]:
            continue
        if only_child and list(cls.subclasses()):
            # if class has a descendant, ignore to have only last "leaves" when only_child is True
            continue
        all_relations = list(cls.INDIRECT_is_a) + list(
            cls.INDIRECT_equivalent_to)  # INDIRECT is to get restrictions from ancestors
        for relation in all_relations:
            cls_has_prop = False
            for descendant in object_descendants:
                # if a subclass of object is the subject of the relation, then object is also
                # the subject of the relation.
                cls_has_prop = check_relation(relation, property, descendant)
                if cls_has_prop:
                    break

            if cls_has_prop:
                # the relation can only be found once, so we exit the loop if found.
                has_prop.append(cls)
                break
    return has_prop


def get_class_leaves(onto_class: owl.entity.ThingClass,
                     max_level: int = -1,
                     current_level: int = 0,
                     return_depth: bool = False) -> list:
    """
    Explores the subclasses of the given class and returns the lowest level subclasses, those
    that have no subclass. If max_level is reached, the subclasses that have this depth are returned.

    :param onto_class: The parent class.
    :param max_level: Maximum level of depth to explore before stopping.
    :param current_level: Used for the recursion stopping condition.
    :param return_depth: Whether to return the depth of each class along with the class.
    :return: A list of the lowest level subclasses, along with their level if return_depth is True.
    """
    # Recursive function, stop if class has no subclass or max level is reached.
    leaf = [onto_class]  # put into iterable, so it can be iterated on during the recursion
    subclasses = list(onto_class.subclasses())
    if not subclasses or (0 <= max_level <= current_level):
        if return_depth:
            return [(leaf[0], current_level)]  # leaf contains only one class
        else:
            return leaf
    else:
        leaves = []
        for subclass in onto_class.subclasses():
            leaves += get_class_leaves(subclass, max_level=max_level, current_level=current_level + 1,
                                       return_depth=return_depth)

        return leaves


def get_property_ranges(property: owl.prop.ObjectPropertyClass,
                        depth: int = 1) -> list[owl.entity.ThingClass]:
    """
    Get all classes that are ranges of an object property. Includes all subclasses of a
    class that is a range of the property.

    :param property: The object property to find the ranges of.
    :param depth: The maximum depth of subclasses to explore.
    :return: A list of all classes and their subclasses that are range of the object property.
    """
    ranges = property.range
    res = []
    for range_class in ranges:
        res += get_class_leaves(range_class, max_level=depth, return_depth=False)
    return res


def get_class_properties(ontology: owl.namespace.Ontology,
                         main_property: owl.prop.ObjectPropertyClass,
                         ignore_classes=None) -> dict[str, dict[str, list[str]]]:
    """
    Finds every subproperty of main_property and gets all their ranges, including subclasses of main ranges.
    Then, for all ranges of one property, finds all classes that have the relation property.Restriction(range).

    :param ontology: The ontology to explore.
    :param main_property: The parent property, only the subproperty of this property will be explored.
    :param ignore_classes: Classes to be ignored when getting class properties.
    :return: A dictionary associating each property with their ranges and maps these ranges with a list of class
             that are defined by the relation property.Restriction(range).
    """
    if ignore_classes is None:
        ignore_classes = []
    object_properties_dict = {}

    for op in main_property.subclasses():
        object_properties_dict[op.name] = {}
        property_ranges = get_property_ranges(op, depth=-1)
        for range_class in property_ranges:
            triplets = get_classes_triplet(ontology, op, range_class, ignore_classes=ignore_classes)
            object_properties_dict[op.name][range_class.name] = [cls.name for cls in triplets]

    return object_properties_dict


def get_classes_names_dict(ontology: owl.namespace.Ontology) -> dict[str, owl.entity.ThingClass]:
    """
    Maps the name of each class with its reference in the ontology.

    :param ontology: The ontology to explore.
    :return: A dictionary mapping the class's name with its owlready reference.
    """
    classes = ontology.classes()
    classes_names_dict = {cls.name: cls for cls in classes}
    return classes_names_dict


def get_object_properties_names_dict(ontology: owl.namespace.Ontology) -> dict[str, owl.prop.ObjectPropertyClass]:
    """
    Maps the name of each object property with its reference in the ontology.

    :param ontology: The ontology to explore.
    :return: A dictionary mapping the object property's name with its owlready reference.
    """
    object_properties = ontology.object_properties()
    object_properties_names_dict = {op.name: op for op in object_properties}
    return object_properties_names_dict


def get_object_property_by_name(ontology: owl.namespace.Ontology, object_property_name: str) -> owl.ObjectPropertyClass:
    """
    Returns the ObjectPropertyClass from an ontology from its name.
    :param ontology: The ontology to explore to get the object property
    :param object_property_name: The name of the object property to find
    :return: The ObjectProperty with the owlready2 format.
    """

    object_properties_name_dict = get_object_properties_names_dict(ontology)
    return object_properties_name_dict[object_property_name]


def get_class_individual(cls: owl.entity.ThingClass):
    """
    Finds one instance of a given class. If no instance exists, creates one.

    :param cls: The class to find an instance of.
    :return: An instance of the given class.
    """
    instances = cls.instances()
    for instance in instances:
        if instance.is_a == [cls]:
            return instance
    indiv_name = cls.name.lower()
    indiv = cls(indiv_name)
    return indiv


def create_class_individual(cls: owl.entity.EntityClass):
    """
    Creates an instance of a given class.
    :param cls: The class to create an instance of
    :return: An instance of the given class
    """

    indiv_name = cls.name.lower()
    indiv = cls(indiv_name)
    return indiv


def add_relation_to_indiv(indiv, object_property: owl.prop.ObjectPropertyClass, object_instance) -> None:
    """
    Adds relation (indiv object_property object_instance) to an individual of the ontology.

    :param indiv: An individual of the ontology.
    :param object_property: An object property of the ontology.
    :param object_instance: Another individual which is an instance of a range class of object property.
    """
    # It is not possible to access object_property list of individual by doing indiv.object_property
    # Thus we use the __getattr__ with the name of the object property to add the new relation.
    # If there was no relation of that object property, __getattr__ returns empty list so append will always work.
    if is_functional(object_property):
        setattr(indiv, object_property.name, object_instance)
    else:
        indiv.__getattr__(object_property.name).append(object_instance)
    # try:
    #     indiv.__getattr__(object_property.name).append(object_instance)
    # except AttributeError:
    #     setattr(indiv, object_property.name, object_instance)


def get_object_instance_from_triplet(indiv, object_property: owl.prop.ObjectPropertyClass,
                                     subject_class: owl.entity.ThingClass):
    """
    Finds the instance of subject_class that is subject of the relation (indiv object_property instance_to_find).
    If no such instance is found, return None.

    :param indiv: The individual that has the wanted relation.
    :param object_property: The object property of the wanted relation.
    :param subject_class: The class of the instance subject of the wanted relation.
    :return: An instance of subject_class that is the subject of the given relation. None if no instance is found.
    """
    subject_instances = indiv.__getattr__(object_property.name)
    if subject_instances:
        if is_functional(object_property):
            return subject_instances
        else:
            for subject_instance in subject_instances:
                if subject_instance in subject_class.instances():
                    return subject_instance
        return None
    else:
        return None


def remove_relation_from_indiv(indiv, object_property: owl.prop.ObjectPropertyClass, object_instance) -> None:
    """
    Removes the relation (indiv object_property object_instance) from individual indiv.
    If relation does not exist, will raise an error.

    :param indiv: The individual to remove the relation from.
    :param object_property: The object property of the relation to remove.
    :param object_instance: The instance, subject of the relation to remove.
    """
    # print(f"Remove {object_property.name} {object_instance}")
    if is_functional(object_property):
        setattr(indiv, object_property.name, None)
    else:
        indiv.__getattr__(object_property.name).remove(object_instance)
    # print(indiv.__getattr__(object_property.name))


def is_consistent(ontology: owl.namespace.Ontology, debug: int = 0, return_explanations: bool = False) \
        -> Union[bool, tuple[bool, str]]:
    """
    Checks whether the ontology is consistent.

    :param ontology: The ontology to check the consistency of.
    :param debug: 0 to give no information about the reasoning, 1 to have limited information and >=2 to have extensive
    explanations.
    :param return_explanations: Will extract explanations from reasoner and output them as raw text. If set to yes,
    it automatically sets debug to 2.
    :return: True if ontology is consistent, False otherwise. If return_explanations is True, returns explanation text
    from the reasoner.
    """
    if return_explanations:
        debug = 2
    temp_onto = owl.get_ontology("http://temp.owl")
    with temp_onto:
        try:
            sync_reasoner_pellet([ontology], infer_property_values=True, debug=debug, apply_results=False)
            if return_explanations:
                return True, ""
            else:
                return True
        except owl.OwlReadyInconsistentOntologyError as e:
            if return_explanations:
                explanations = extract_explanations_from_reasoner(e)
                return False, explanations
            elif debug >= 2:
                print(e)
            return False


def extract_explanations_from_reasoner(reasoner_output):
    reasoner_output = str(reasoner_output)
    expl_start = reasoner_output.find("1)")
    expl_end = reasoner_output.find("\n\n\n")
    explanations = reasoner_output[expl_start + 3:expl_end]
    return explanations


def is_functional(property: owl.prop.ObjectPropertyClass) -> bool:
    return owl.FunctionalProperty in property.is_a


def is_functional_by_name(ontology: owl.namespace.Ontology, property_name: str) -> bool:
    object_properties_dict = get_object_properties_names_dict(ontology)
    return is_functional(object_properties_dict[property_name])


def get_class_parents(classes: Union[Iterable[owl.entity.ThingClass], owl.entity.ThingClass]) -> set[owl.EntityClass]:
    """
    Returns a set of the direct parents of a list of classes.
    Inspired from owlready2 code to get a class ancestors.
    :param classes: List of classes to get the direct parents from.
    :return: A set of direct parents.
    """
    if not isinstance(classes, Iterable):
        # We assume it is a unique element and convert it to a list
        classes = [classes]
    s = set()
    for cls in classes:
        for parent in cls.__bases__:
            if isinstance(parent, owl.EntityClass):
                s.add(parent)
    return s


def get_all_descendants(classes: Union[Iterable[owl.entity.ThingClass], owl.entity.ThingClass]):
    """
    Get every descendant of a combination of classes. Equivalent to class.descendants() if classes only has one element.
    :param classes: List of classes to get all the descendants from.
    :return: Set of all descendants
    """
    if not isinstance(classes, Iterable):
        classes = [classes]
    s = set()
    for cls in classes:
        s.update(cls.descendants())
    return s


def get_class_descendants(ontology: owl.namespace.Ontology, cls: owl.entity.ThingClass) -> set[owl.EntityClass]:
    """
    Returns a set of the direct descendants of a class.
    Inspired from owlready2 code to get a class descendants.
    :param ontology: The ontology to get the descendants from.
    :param cls: The class to get the direct descendants from.
    :return: A set of direct descendants.
    """
    s = set()
    world = owl.default_world
    for x in world._get_obj_triples_po_s(cls._rdfs_is_a, cls.storid):
        if not x < 0:
            descendant = world._get_by_storid(x, None, cls.__class__, ontology)
            if descendant is cls:
                continue
            if descendant not in s:
                s.add(descendant)
    return s


def create_individual_from_assertions(assertions: list[tuple[owl.ObjectPropertyClass, owl.entity]], default_class,
                                      is_a=None):
    """
    Creates a new individual with a given list of assertions.

    :param assertions:
    :param is_a:
    :param default_class:
    :return:
    """
    if is_a is None:
        is_a = []
    new_individual = default_class("indiv")
    new_individual.is_a = is_a
    for assertion in assertions:
        property = assertion[0]
        instance = assertion[1]
        add_relation_to_indiv(new_individual, property, instance)
    return new_individual


def has_one_of(classes: list[owl.ThingClass]) -> list[owl.OneOf]:
    """
    Checks if a class is defined by a OneOf construction and returns the list of the OneOf definitions.
    :param classes: Classes to check
    :return: List of OneOf definitions, empty list of none found.
    """
    definitions = []
    for cls in classes:
        definitions += cls.is_a + cls.equivalent_to
    one_ofs = []
    for definition in definitions:
        if isinstance(definition, owl.OneOf):
            one_ofs.append(definition)
    return one_ofs


def is_subproperty_of(child_property: owl.ObjectProperty, parent_property: owl.ObjectProperty) -> bool:
    """
    Checks if the ObjectProperty child_property is subproperty of parent_property.
    :param child_property: Property to check
    :param parent_property: Possible parent property
    :return: True if child_property is subproperty of parent_property, False otherwise
    """
    return parent_property in child_property.is_a
