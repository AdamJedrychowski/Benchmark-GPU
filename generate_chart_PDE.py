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

    plt.xlabel('Number of Time Steps')
    plt.ylabel('Time (s)')
    plt.title('Number of Time Steps vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    files = {
        'Sycl': 'results/FiniteDifferenceMethodsforPDEs/Sycl.txt',
        'CUDA': 'results/FiniteDifferenceMethodsforPDEs/CUDA.txt',
        'OpenCL': 'results/FiniteDifferenceMethodsforPDEs/OpenCL.txt'
    }

    data_dict = {}
    for label, file_path in files.items():
        data_dict[label] = read_data(file_path)

    plot_chart(data_dict)

if __name__ == '__main__':
    main()