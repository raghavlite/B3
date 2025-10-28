import wandb

# Initialize the WandB API
api = wandb.Api()

# Replace with your project path (e.g., "username/project-name")
project_path = "dukeuniversity/VLM2Vec-pro"
runs = api.runs(project_path)
print(runs)
# Iterate over runs and update the display name
for count, run in enumerate(runs):
    new_name = run.display_name
    if(new_name.startswith("./MMEB-trainedmodels/")):
        # # import ipdb; ipdb.set_trace()
        # print(f"Name {run.display_name}", flush=True)
        # print(f"Run Name {run.config['run_name']}", flush=True)
        # print("")
        new_name=new_name.split("/")[-1]
        print(f"Updating run {run.id} to new display name: {new_name}")
        run.display_name = run.display_name.split("/")[-1]
        run.update()
        # if(count==6):
        #     break;
        # run.update(description= new_name)
        # run.config["run_name"] = new_name