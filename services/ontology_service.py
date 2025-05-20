from rdflib import Graph, Namespace

edu = Namespace("http://example.org/edu#")
g = Graph()
g.parse("ontology.ttl", format="turtle")

def get_related_topics(goal: str) -> list[str]:
    """Return prerequisite or enhancing topics based on learner goal."""
    goal_uri = edu[goal.replace(" ", "")]
    related = set()

    # Get prerequisites
    for s in g.subjects(predicate=edu.isPrerequisiteFor, object=goal_uri):
        related.add(str(s).split("#")[-1])

    # Get enhancing skills
    for s in g.subjects(predicate=edu.enhances, object=goal_uri):
        related.add(str(s).split("#")[-1])

    # Get sub-topics if goal is high-level
    for s in g.subjects(predicate=edu.subTopicOf, object=goal_uri):
        related.add(str(s).split("#")[-1])

    return list(related)
