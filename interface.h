#ifndef INTERFACE_H
#define INTERFACE_H
#include<SFML/Graphics.hpp>
#include"core_engine.h"
#include"pipeline.h"
#include<string>
#include<iostream>
#include<deque>
#include<queue>
#include<utility>

class SimulationUI {
private:
    sf::RenderWindow window;
    sf::Font font;
    sf::Text uiText;

    float currentPrediction  = 0.0f;
    float currentTarget      = 0.0f;
    int   currentSampleIndex = 0;
    double bestLoss = 999.0;

    NeuralNetwork& network;
    TrainingData&  data;

    bool   isTraining;
    double learningRate;
    int    currentEpoch;
    double lastLoss;

    // last 60 epoch losses feeds the graph
    std::deque<float> lossHistory;

    // tracks the 3 best epochs, max heap so worst of best 3 sits at top
    using EpochEntry = std::pair<double, int>;
    std::priority_queue<EpochEntry> bestEpochs;
    static const int TOP_K = 3;

    // animation timing, all time based so speed is independent of frame rate
    sf::Clock animClock;
    float     animElapsed;
    int       lastVisualizedSample;
    static constexpr float SECONDS_PER_PHASE = 0.5f;

    // cached sigmoid output per neuron, rebuilt each step so glow reflects actual state
    std::vector<std::vector<float>> nodeActivations;
    void computeActivations();

    void handleInput();
    void render();
    void drawNetworkGraph();
    void drawDashboard();
    void drawSparkline(float x, float y, float width, float height);

public:
    SimulationUI(NeuralNetwork& net, TrainingData& tData);
    void run();
};

#endif
