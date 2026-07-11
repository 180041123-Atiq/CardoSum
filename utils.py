import zipfile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def unzip_file(source_file, out_dir):
    with zipfile.ZipFile(source_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)


def tensorboard_event_reader(file_path):
    
    ea = EventAccumulator(file_path)
    ea.Reload()
    
    eval_loss = ea.Scalars("eval/loss")
    print(eval_loss)

if __name__ == '__main__':
    tensorboard_event_reader('runs\Jul10_19-45-30_c51a494ff5d9\events.out.tfevents.1783712730.c51a494ff5d9.3314.0')
    # pass
    # # print(gen_layout_prompt(106))
    # # plot_graph()
    # unzip_file('checkpoint-720.zip','HVO')