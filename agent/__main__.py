from agent.graph import graph

g = graph.get_graph()

# Mermaid 텍스트 출력
mermaid = g.draw_mermaid()
print("=== Mermaid Graph ===")
print(mermaid)
