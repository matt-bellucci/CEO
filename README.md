# Counterfactual Explanations for Ontologies (CEO)

CEO is a method that generates counterfactuals explanations for OWL ontologies.
It is compatible with any OWL ontology provided in RDF/XML format.

Package requirements are provided in `requirements.txt` and `Pipfile` for pipenv environments.
To generate a counterfactual, execute the `main.py` script.
The generation may take a few minutes.

A survey was conducted to test this method on an ontolgoy of musical instruments, the details are given in the directory `Survey`.

## How to enter your own counterfactual 
The first case of the survey (see directory `Survey`) is the default counterfactual example.
Here is a step-by-step guide to modify the counterfactual.

1. Open the ontology in an ontology editor.
2. Create or edit the `test` individual, without specifying its class, only its Object property assertions.  
3. Add the desired assertions to this individual.
4. Go to `main.py` script.
5. Write the path to the ontology in the variable `onto_path`
6. Give the name of the individual in the variable `original_ikg`, preceded by onto (if the name of the individual is `test`, the variable should take the value `onto.test`).
7. Give the desired class for the counterfactual in the variable `counterfactual_class` in the form `[onto.Class]` where `Class` is the name of a class in the ontology.
8. Optionally, set `display_graph` to `True` or `False` to decide whether to display the graph of generated counterfactuals and how they are connected.
9. Run the script, it may take a few minutes. Each step is printed to follow the generation progression.
10. The graph is displayed in another window if `display_graph` was set to `True`, then a list of valid counterfactuals ranked by proximity is listed.

## Overview of the code

The code uses the Owlready2 library to interact with the ontology.
Several utilities to simplify the use of this library are in the `onto_utils.py` file.
Since only a consistency check is conducted when calling the reasoner, the file `custom_reasoning.py` contains a call to Pellet reasoner that only checks the ontology consistency.
The `graph.py` file contains the definition of classes to represent an individual and different assertion types in Python.
Then, `graph_generator.py` contains all the functions that are used to generate the counterfactuals. It uses the NetworkX library to represent the graph.
