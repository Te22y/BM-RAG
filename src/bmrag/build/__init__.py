"""Build pipeline: from paper PDFs to the paper index, the knowledge graph and its vector index.

Run it with ``python -m bmrag.build`` (see bmrag.build.__main__ and docs/build_pipeline.md).

Paper index (bmrag.build.paper_index), read by the Reasoner's paper search:
  chunk      partition each PDF with unstructured and split it into title-based chunks
  clean      drop references and back matter
  separate   split text, tables and images; filter small images; write the manifest
  summarize  summarize every element with an LLM and embed the summaries

Knowledge graph (bmrag.build.graph) and its index (bmrag.build.stores):
  extract      one multimodal LLM call per PDF returns its cases and rules
  entities     normalize them into entities and relations
  quality      check required fields, labels and relations
  cypher       write the Cypher MERGE statements
  standardize  align entity names and enforce the PCM ontology
  kg-index     embed cases and rules into the case and rule stores
  neo4j        validate the Cypher and optionally import it

Settings that change the output live in bmrag.build.profiles; paper_v1 reproduces the paper's data.
"""
