### **AI Neural Network Simulator**



##### **Overview**

The AI Neural Network Simulator is a custom Multi Layer Perceptron built entirely from scratch in C++17. It features a custom mathematical engine for forward and backward propagation and a high performance SFML 3 dashboard to visualize the training process in real time.



##### **Key Features**

Custom Mathematical Engine: Built in matrix algebra handling dot products, transpositions and the chain rule for backpropagation across Dense and Sigmoid layers.



Real Time Dashboard: A visual interface showing data pulses, live inferences, dynamic semantic grading, a real time area chart for loss history and epochs per second telemetry.



Robust Data Pipeline: Custom CSV parser for the heart disease dataset, Xavier weight initialization and binary model state saving and loading.



##### **File Structure**

main.cpp: Entry point that loads data, builds the 11 to 16 to 8 to 1 network architecture, and launches the UI.



core\_engine: The math heart containing Matrix and Layer classes.



pipeline: Handles data ingestion, weight initialization and file I/O.



interface: Manages the SFML window, input handling, and rendering logic.



##### **Build Instructions**

This project requires a C++17 compiler, CMake version 3.10 or higher and SFML 3.



First, generate the build files by running: cmake -S . -B build



Second, compile the executable by running: cmake --build build



Finally, run the simulation by executing the compiled main.exe file inside the build folder.



##### **Controls**

SPACE: Toggle Training (Pause or Resume)



UP ARROW: Increase Learning Rate



DOWN ARROW: Decrease Learning Rate



S KEY: Save current model weights

