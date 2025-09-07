import matplotlib.pyplot as plt
import csv

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            data.append((float(row[0]), float(row[1])))
    return data

def plot_chart(data_dict):
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        x, y = zip(*data)
        plt.plot(x, y, label=label)

    plt.xlabel('Table Size (N)')
    plt.ylabel('Time (s)')
    plt.title('Table Size vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    files = {
        'Sycl': 'results/MapReduce/Sycl.txt',
        'CUDA': 'results/MapReduce/CUDA.txt',
        'OpenCL': 'results/MapReduce/OpenCL.txt'
    }

    data_dict = {}
    for label, file_path in files.items():
        data_dict[label] = read_data(file_path)

    plot_chart(data_dict)

if __name__ == '__main__':
    main()