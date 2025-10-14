from nntool.experiment import read_toml_file, get_output_path

env = read_toml_file("./env.toml")

project_path = env["project"]["path"]

dataset_path = f"{project_path}/datasets"

project_output_path = f"{project_path}/output"

output_path, _ = get_output_path(project_output_path)
