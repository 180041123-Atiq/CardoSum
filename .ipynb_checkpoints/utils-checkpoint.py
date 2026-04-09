import zipfile


def unzip_file(source_file, out_dir):
    with zipfile.ZipFile(source_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

if __name__ == '__main__':
    pass
    # print(gen_layout_prompt(106))
    # plot_graph()
    unzip_file('checkpoint-720.zip','HVO')