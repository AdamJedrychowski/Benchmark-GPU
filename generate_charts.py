import matplotlib.pyplot as plt
import csv
import argparse
import sys
import os

def read_data(file_path):
    """Read data from CSV file with semicolon delimiter."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return data
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if len(row) >= 2:
                try:
                    data.append((float(row[0]), float(row[1])))
                except ValueError:
                    print(f"Warning: Could not parse row {row} in {file_path}")
    return data

def plot_chart(data_dict, chart_config):
    """Plot chart with given data and configuration."""
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        if data:  # Only plot if data exists
            x, y = zip(*data)
            plt.plot(x, y, label=label, marker='o')

    plt.xlabel(chart_config['xlabel'])
    plt.ylabel('Time (s)')
    plt.title(chart_config['title'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_chart_config(chart_type):
    """Get configuration for different chart types."""
    configs = {
        'jacobi': {
            'folder': 'Jacobi',
            'title': 'Matrix Size vs Time (Jacobi Method)',
            'xlabel': 'Matrix Size (NxN)',
            'description': 'Jacobi iterative method performance'
        },
        'montecarlo': {
            'folder': 'MonteCarlo',
            'title': 'Number of Points vs Time (Monte Carlo)',
            'xlabel': 'Number of Points (N)',
            'description': 'Monte Carlo simulation performance'
        },
        'mapreduce': {
            'folder': 'MapReduce',
            'title': 'Table Size vs Time (MapReduce)',
            'xlabel': 'Table Size (N)',
            'description': 'MapReduce operation performance'
        },
        'pde': {
            'folder': 'FiniteDifferenceMethodsforPDEs',
            'title': 'Number of Time Steps vs Time (PDE)',
            'xlabel': 'Number of Time Steps',
            'description': 'Finite Difference Methods for PDEs performance'
        }
    }
    return configs.get(chart_type.lower())

def generate_chart(chart_type, frameworks=None):
    """Generate chart for specified type and frameworks."""
    config = get_chart_config(chart_type)
    if not config:
        print(f"Error: Unknown chart type '{chart_type}'")
        print("Available chart types: jacobi, montecarlo, mapreduce, pde")
        return False

    # Default frameworks if none specified
    if frameworks is None:
        frameworks = ['Sycl', 'CUDA', 'OpenCL', 'OpenACC']

    # Build file paths
    files = {}
    for framework in frameworks:
        file_path = f"results/{config['folder']}/{framework}.txt"
        files[framework] = file_path

    # Read data
    data_dict = {}
    for label, file_path in files.items():
        data = read_data(file_path)
        if data:
            data_dict[label] = data
        else:
            print(f"No data found for {label}")

    if not data_dict:
        print(f"Error: No data found for any framework in {chart_type}")
        return False

    # Plot chart
    plot_chart(data_dict, config)
    print(f"Generated chart for {config['description']}")
    return True

def list_available_charts():
    """List all available chart types."""
    print("Available chart types:")
    for chart_type in ['jacobi', 'montecarlo', 'mapreduce', 'pde']:
        config = get_chart_config(chart_type)
        print(f"  {chart_type:<12} - {config['description']}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate performance charts for GPU benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python generate_charts.py jacobi
  python generate_charts.py montecarlo --frameworks CUDA Sycl OpenACC
  python generate_charts.py pde --frameworks OpenCL OpenACC
  python generate_charts.py --list
  python generate_charts.py --all
        '''
    )
    
    parser.add_argument(
        'chart_type',
        nargs='?',
        help='Type of chart to generate (jacobi, montecarlo, mapreduce, pde)'
    )
    
    parser.add_argument(
        '--frameworks',
        nargs='+',
        default=['Sycl', 'CUDA', 'OpenCL', 'OpenACC'],
        help='Frameworks to include in chart (default: Sycl CUDA OpenCL OpenACC)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available chart types'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all chart types'
    )

    args = parser.parse_args()

    if args.list:
        list_available_charts()
        return

    if args.all:
        chart_types = ['jacobi', 'montecarlo', 'mapreduce', 'pde']
        success_count = 0
        for chart_type in chart_types:
            print(f"\nGenerating {chart_type} chart...")
            if generate_chart(chart_type, args.frameworks):
                success_count += 1
        print(f"\nSuccessfully generated {success_count}/{len(chart_types)} charts")
        return

    if not args.chart_type:
        parser.print_help()
        print("\nUse --list to see available chart types")
        return

    if not generate_chart(args.chart_type, args.frameworks):
        sys.exit(1)

if __name__ == '__main__':
    main()
