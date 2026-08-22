import pathlib
p = pathlib.Path("openevolve/differentiable_architecture_search.py")
s = p.read_text()
s = s.replace("        output_dim=output_dim if False else output_dim,", "        output_dim=output_dim,")
assert "if False" not in s, "if False still present"
p.write_text(s)
print("cleaned")
