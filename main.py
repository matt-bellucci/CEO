from onto_utils import load_ontology
from graph_generator import test_counterfactuals

onto_path = "instruments.owl"
onto = load_ontology(onto_path)
original_ikg = onto.test
counterfactual_class = [onto.Harpsichord]
# display graph shows the part of the graph explored, with the closest IKG in orange.
test_counterfactuals(onto, original_ikg, counterfactual_class, display_graph=True)
