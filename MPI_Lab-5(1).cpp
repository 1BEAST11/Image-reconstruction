#include <iostream>
#include <vector>
#include <mpi.h>
#include <random>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<pair<int, int>> cell_options(int size, int x, int y, vector<int>& image)
{
    vector<pair<int, int>> options; // вектор с вариантами клеток для перехода
    // #1
    if (x - 1 >= 0 && y - 1 >= 0)
    {
        if (image[(x - 1) * size + (y - 1)] < 255)
        {
            options.emplace_back(x - 1, y - 1); // левый нижний угол
        }
    }
    // #2
    if (y - 1 >= 0)
    {
        if (image[x * size + (y - 1)] < 255)
        {
            options.emplace_back(x, y - 1); // середина внизу
        }
    }
    // #3
    if (x + 1 < size && y - 1 >= 0)
    {
        if (image[(x + 1) * size + (y - 1)] < 255)
        {
            options.emplace_back(x + 1, y - 1); // правый нижний угол
        }
    }
    // #4
    if (x + 1 < size)
    {
        if (image[(x + 1) * size + y] < 255)
        {
            options.emplace_back(x + 1, y); // середина вверху
        }
    }
    // #5
    if (x + 1 < size && y + 1 < size)
    {
        if (image[(x + 1) * size + (y + 1)] < 255)
        {
            options.emplace_back(x + 1, y + 1); // правый верхний угол
        }
    }
    // #6
    if (y + 1 < size)
    {
        if (image[x * size + (y + 1)] < 255)
        {
            options.emplace_back(x, y + 1); // середина справа
        }
    }
    // #7
    if (x - 1 >= 0 && y + 1 < size)
    {
        if (image[(x - 1) * size + (y + 1)] < 255)
        {
            options.emplace_back(x - 1, y + 1); // левый верхний угол
        }
    }
    // #8
    if (x - 1 >= 0)
    {
        if (image[(x - 1) * size + y] < 255)
        {
            options.emplace_back(x - 1, y); // середина слева
        }
    }

    return options;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = 0.0; // начальное время
    double endTime = 0.0; // конечное время

    int image_size = 100;
  
    vector<int> image(image_size * image_size, 0);
    vector<int> target(image_size * image_size, 0);
    
    int target_sum = 0; // сумма значений целевого изображения
    int image_sum = 0; // сумма значений текущего изображения

    // преобразование начального изображения в массив
    Mat picture = imread("fox.jpg", IMREAD_GRAYSCALE);

    for (int i = 0; i < picture.rows; ++i)
    {
        for (int j = 0; j < picture.cols; ++j)
        {
            uchar pixel = picture.at<uchar>(i, j);
            target[i * image_size + j] = 255 - static_cast<int>(pixel);
            target_sum += target[i * image_size + j];
        }
    }

    if (rank == 0)
    {
        cout << "Target image sum: " << target_sum << endl;

        // измерение начала времени параллельной части
        startTime = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    random_device rd;
    mt19937 generator(rd() + rank); // генератор случайных чисел
    uniform_int_distribution<int> distribution(0, image_size - 1);// создание равномерного распределения случайных чисел в заданном диапазоне
    
    int x, y; // координаты изображения
    double norm = 0.0;
    int index = 0;
    double delta = 0.0;

    x = distribution(generator);
    y = distribution(generator);
    index = x * image_size + y;

    vector<int> gathered_index(size);
    vector<int> gathered_check_index(size);

    MPI_Gather(&index, 1, MPI_INT, gathered_index.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < gathered_index.size(); i++)
        {
            image[gathered_index[i]]++;
            image_sum++;
        }

        norm = target_sum / image_sum;
        delta = 0;

        for (int i = 0; i < image.size(); i++)
        {
            delta += max(0.0, target[i] - image[i] * norm);
        }
    }

    // Рассылка обновленного изображения всем процессам
    MPI_Bcast(image.data(), image_size * image_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // цикл приведения изображения к эталонному виду
    int k = 0;
    double previous_delta = 0.0;
    int check_index = 0;

    while (delta > target_sum * 0.1)
    {
        if (k % 1000 == 0)
        {
            if (rank == 0)
            {
                cout << k << ") " << delta << endl;

                /*if (k % 5000 == 0)
                {
                    string name = to_string(k) + ".txt";
                    ofstream file(name);

                    for (int i = 0; i < image_size; i++)
                    {
                        for (int j = 0; j < image_size; j++)
                        {
                            file << 255 - image[i * image_size + j] << " ";
                        }
                        file << endl;
                    }
                }*/
            }
        }

        // доступные варианты перемещения
        vector<pair<int, int>> options = cell_options(image_size, x, y, image);

        pair<int, int> best_cell; // лучшая клетка для перехода
        vector <int> cell_var; // приоритет лучшей клетки

        // цикл прохода по всем вариантам
        for (const auto& cell : options)
        {
            int image_cell = image[cell.first * image_size + cell.second];
            int target_cell = target[cell.first * image_size + cell.second];

            int trial = 0;
            trial = target_cell - image_cell * norm;

            cell_var.push_back(trial);
        }

        if (!cell_var.empty())
        {
            // выбор клетки с максимальным недобором
            auto max_iter = max_element(cell_var.begin(), cell_var.end());
            int choice = distance(cell_var.begin(), max_iter);

            // координаты выбранной клетки
            best_cell = options[choice];
            x = best_cell.first;
            y = best_cell.second;
        }

        index = x * image_size + y;
        MPI_Gather(&index, 1, MPI_INT, gathered_index.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 0; i < gathered_index.size(); i++)
            {
                image[gathered_index[i]]++;
                image_sum++;
            }
       
            norm = target_sum / image_sum;
            delta = 0;

            for (int i = 0; i < image.size(); i++)
            {
                delta += max(0.0, target[i] - image[i] * norm);
            }
        }

        // Рассылка обновленного изображения всем процессам
        MPI_Bcast(image.data(), image_size * image_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        k++;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        endTime = MPI_Wtime(); // измерение времени конца параллельной части
        double parallelExecutionTime = endTime - startTime; // расчет общего времени паралельной части
        cout << "Time of the parallel part: " << parallelExecutionTime << " seconds" << endl;
        cout << "Final delta: " << delta << endl;
        cout << "Current image sum: " << image_sum << endl;
        cout << "Number of iterations: " << k << endl;

        ofstream file("final.txt");

        for (int i = 0; i < image_size; i++)
        {
            for (int j = 0; j < image_size; j++)
            {
                file << 255 - image[i * image_size + j] << " ";
            }
            file << endl;
        }
    }

    MPI_Finalize();

    return 0;
}