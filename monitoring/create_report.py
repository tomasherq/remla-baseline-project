import markdown
import pdfkit
import json
import os
import sys
import json
import matplotlib.pyplot as plt


# Get this location to properly link the images and all the files
CURRENT_LOCATION = ".."+os.getcwd()+"/monitoring"


for i in range(50):
    CURRENT_LOCATION = "../"+CURRENT_LOCATION

LOCATION_GRAPH = CURRENT_LOCATION+"/resources/chart_usage.svg"


def create_chart(location_save):
    MARGIN_RAM = 10
    MARGIN_DISK = 4

    with open("monitoring/metrics/results_execution.json", "r") as file_read:
        results = json.load(file_read)

    def get_y_ticks_io(value):
        starting_tick = int(value*0.1)
        starting_tick -= starting_tick % 5000
        tick = starting_tick
        ticks = list()
        labels = list()
        while value > tick:
            tick_text = str(int(tick/1000))+"K"
            labels.append(tick_text)
            ticks.append(tick)
            tick += starting_tick
        return ticks, labels

    f = plt.figure()
    f.set_figwidth(35)
    f.set_figheight(15)

    # using rc function
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('xtick', labelsize=12)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=15)  # fontsize of the legend
    # This is the legend used by everyone
    x = list()
    for i in range(len(results["cpu_usage"])):
        x.append(0.5*i)

    # CPU
    y = results

    plt.subplot(2, 2, 1)
    plt.ylabel('Usage (%)')
    plt.ylim(0, 102)
    plt.xlim(0, len(results)*0.1)

    plt.title('CPU')

    x_tics = [tic for tic in x.copy() if tic % 10 == 0]
    x_tics.pop(0)

    plt.xticks(x_tics)
    plt.plot(x, y["cpu_usage"], label="", linewidth=2, markersize=0)

    plt.subplot(2, 2, 2)
    plt.title('Memory')

    limit_memory_lower = min([min(y["ram_memory"]), min(y["swap_memory"])])
    if limit_memory_lower > MARGIN_RAM:
        limit_memory_lower -= MARGIN_RAM
    else:
        limit_memory_lower = -1

    limit_memory_higher = max([max(y["ram_memory"]), max(y["swap_memory"])])

    if limit_memory_higher < 100+MARGIN_RAM:
        limit_memory_higher += MARGIN_RAM
    else:
        limit_memory_higher = 100

    plt.ylim(limit_memory_lower, limit_memory_higher)
    plt.xlim(0, len(results)*0.1)
    plt.xticks(x_tics)

    plt.plot(x, y["ram_memory"], label="RAM", linewidth=3, markersize=0)
    plt.plot(x, y["swap_memory"], label="Swap", linewidth=3, markersize=0)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Disk usage')

    limit_disk_low = min(y["disk_usage"])

    if limit_disk_low-MARGIN_DISK > 0:
        limit_disk_low = limit_disk_low-MARGIN_DISK
    else:
        limit_disk_low = 0

    limit_disk_high = min(y["disk_usage"])

    if limit_disk_high+MARGIN_DISK < 100:
        limit_disk_high = limit_disk_high+MARGIN_DISK
    else:
        limit_disk_high = 100

    plt.ylim(limit_disk_low, limit_disk_high)
    plt.xlim(0, len(results)*0.1)
    plt.xticks(x_tics)
    plt.ylabel('Usage (%)')
    plt.xlabel('Time (s)')

    plt.plot(x, y["disk_usage"], label="", linewidth=3, markersize=0)

    plt.subplot(2, 2, 4)
    plt.title('Disk I/O')

    limit_disk_low = 0
    limit_disk_high = max([max(y["reads"]), max(y["writes"])])

    plt.ylim(limit_disk_low, limit_disk_high)
    plt.xlim(0, len(results)*0.1)
    plt.xticks(x_tics)
    yticks, ylabels = get_y_ticks_io(int(limit_disk_high))
    plt.yticks(yticks, ylabels)

    plt.ylabel('Nº of operations')
    plt.xlabel('Time (s)')

    plt.plot(x, y["reads"], label="Reads", linewidth=3, markersize=0)
    plt.plot(x, y["writes"], label="Writes", linewidth=3, markersize=0)

    plt.legend()

    plt.savefig(location_save, bbox_inches="tight")


def read_template_markdown():
    with open("monitoring/resources/report_template.md", "r") as file:

        return file.read()


def read_json(location):
    with open(location, "r") as file:
        return json.load(file)


def get_size_of_files():

    # The user can input the location of the datasets used for the model
    # we default to the ones provided by the TA's
    location = "data/"
    if len(sys.argv) > 1:
        location = sys.argv[1]
    else:
        print("No path to datafiles inserted in arguments (first one), used default path data/")

    size_files = {}

    for file in os.listdir(location):
        full_path = location+file
        name = file.split(".")[0]
        # Size in Mb
        size_files[name] = float(os.path.getsize(full_path)/2**20)

    return size_files


def read_run_times():
    runtimes = {}
    start_time = 0
    total_time = 0

    with open(f'{CURRENT_LOCATION}/resources/runtimes.txt', "r") as file:
        for line in file:
            line = line.lower().strip()
            data = line.split(";")

            if "start" in data:
                start_time = float(data[1])
            else:

                runtimes[data[0]] = float(data[1])-start_time
                total_time += runtimes[data[0]]

    runtimes["total"] = total_time
    return runtimes


def replace_in_template(data, output_document):

    for key, value in data.items():
        selector = f'##{key}##'
        output_document = output_document.replace(selector, str(round(value, 2)))

    return output_document


def get_metrics_models():
    models_used = ["bag", "tfidf"]
    metrics_models = {}

    for model_used in models_used:
        metrics = read_json(f"results/metrics-{model_used}.json")
        for metric, value in metrics.items():
            metrics_models[metric+model_used] = float(value)

    return metrics_models


def get_best_model(metrics):

    diff_metrics = 0
    for metric in metrics:

        if "bag" in metric:

            diff_metrics = metrics[metric]-metrics[metric.replace("bag", "tfidf")]

        else:
            break
    best_model = "Bag of Words"

    if diff_metrics < 0:
        best_model = "TF-IDF"
    elif diff_metrics == 0:
        best_model = "is none, they both performed the same"

    return best_model


# Crete chart for the basic resource metrics
create_chart(LOCATION_GRAPH)

# Change the location of the image in the template
template = read_template_markdown()


output_document = template.replace("##LocationGraph##", LOCATION_GRAPH)

# Input the metrics recorded for the basic monitoring
metrics_resources = read_json(CURRENT_LOCATION+"/metrics/results_execution.json")["total"]

output_document = replace_in_template(metrics_resources, output_document)

# Input the run times of the different steps
runtimes = read_run_times()

output_document = replace_in_template(runtimes, output_document)

# Input the size of the datasets used in the run
size_of_files = get_size_of_files()

output_document = replace_in_template(size_of_files, output_document)

# Input the metrics for the models tested
metrics_models = get_metrics_models()

output_document = replace_in_template(metrics_models, output_document)


if metrics_models["accuracybag"] < metrics_models["accuracytfidf"]:
    best_model = "TF-IDF"

output_document = output_document.replace("##bestmodel##", get_best_model(metrics_models))

# Add this HTML to style the tables of the report
source_html = markdown.markdown(output_document, extensions=['markdown.extensions.tables'])

source_html = """<style>
img{
    max-width:100%;
}
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
td,th {
    padding-right:10px;
    padding-left:10px;
}
</style>"""+source_html

# We print the pdf

output_filename = "run_report.pdf"

pdfkit.from_string(source_html, output_filename)

print(f"The report can be found at the location {os.getcwd()}/{output_filename}")
