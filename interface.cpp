#include"interface.h"
#include<sstream>
#include<iomanip>
#include<cmath>
#include<cstdint>
#include<algorithm>

SimulationUI::SimulationUI(NeuralNetwork& net, TrainingData& tData)
    : network(net), data(tData), isTraining(false), learningRate(0.01),
      currentEpoch(0), lastLoss(0.0), uiText(font),
      animElapsed(0.f), lastVisualizedSample(-1) {

    window.create(sf::VideoMode({1200, 800}), "AI Neural Network Simulator");
    window.setFramerateLimit(60);

    if (!font.openFromFile("arial.ttf")) {
        std::cerr << "ERROR: Could not load arial.ttf." << std::endl;
        exit(1);
    }

    uiText.setFont(font);
}

void SimulationUI::run() {
    while (window.isOpen()) {
        handleInput();

        float dt = animClock.restart().asSeconds();

        // run one epoch if training is on
        if (isTraining && data.size() > 0) {
            std::vector<double> losses = TrainingPipeline::train(network, data, 1, learningRate);

            if (!losses.empty()) {
                lastLoss = losses.back();
                if (lastLoss < bestLoss) bestLoss = lastLoss;

                bestEpochs.push(EpochEntry(lastLoss, currentEpoch + 1));
                if (static_cast<int>(bestEpochs.size()) > TOP_K) bestEpochs.pop();

                lossHistory.push_back(static_cast<float>(lastLoss));
                if (lossHistory.size() > 60) lossHistory.pop_front();
            }
            currentEpoch++;
        }

        // input col + one col per dense layer
        int numColumns = 1;
        for (const auto& l : network.get_layers())
            if (dynamic_cast<DenseLayer*>(l.get())) numColumns++;

        const float holdAfterComplete = 0.4f;
        const float fullDuration = SECONDS_PER_PHASE * numColumns + holdAfterComplete;

        // pick a random patient, run it through the network, reset the wave
        auto pickNewSample = [&]() {
            currentSampleIndex = std::rand() % data.size();
            const TrainingSample& sample = data.get_samples()[currentSampleIndex];
            Matrix guess = network.predict(sample.input);
            currentPrediction = static_cast<float>(guess.data[0][0]);
            currentTarget     = static_cast<float>(sample.target.data[0][0]);
            animElapsed = 0.f;
            lastVisualizedSample = currentSampleIndex;
            computeActivations();
        };

        // show something on screen before the user hits space
        if (lastVisualizedSample == -1 && data.size() > 0)
            pickNewSample();

        if (isTraining && data.size() > 0) {
            animElapsed += dt;
            if (animElapsed >= fullDuration)  // wave finished, move to next patient
                pickNewSample();
        }

        render();
    }
}

void SimulationUI::computeActivations() {
    nodeActivations.clear();
    if (currentSampleIndex < 0 || data.size() == 0) return;

    const TrainingSample& sample = data.get_samples()[currentSampleIndex];
    Matrix current = sample.input;

    // column 0 = raw inputs
    std::vector<float> col;
    col.reserve(current.cols);
    for (int j = 0; j < current.cols; j++)
        col.push_back(static_cast<float>(current.data[0][j]));
    nodeActivations.push_back(col);

    // only grab values after sigmoid layers, those are the actual activations
    const auto& layers = network.get_layers();
    for (const auto& layer : layers) {
        current = layer->forward(current);
        if (layer->get_type() == "Sigmoid") {
            col.clear();
            col.reserve(current.cols);
            for (int j = 0; j < current.cols; j++)
                col.push_back(static_cast<float>(current.data[0][j]));
            nodeActivations.push_back(col);
        }
    }
}

void SimulationUI::handleInput() {
    while (const std::optional<sf::Event> event = window.pollEvent()) {
        if (event->is<sf::Event::Closed>())
            window.close();

        if (const auto* resized = event->getIf<sf::Event::Resized>()) {
            sf::FloatRect visibleArea({0.f, 0.f}, sf::Vector2f(resized->size));
            window.setView(sf::View(visibleArea));
        }

        if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
            switch (keyPressed->code) {
                case sf::Keyboard::Key::Space:
                    if (data.size() > 0) isTraining = !isTraining;
                    break;
                case sf::Keyboard::Key::Up:
                    learningRate += 0.01;
                    break;
                case sf::Keyboard::Key::Down:
                    if (learningRate > 0.01) learningRate -= 0.01;
                    break;
                case sf::Keyboard::Key::S:
                    ModelIO::save_model(network, "network_save.dat");
                    break;
                default:
                    break;
            }
        }
    }
}

void SimulationUI::render() {
    window.clear(sf::Color(22, 24, 28));
    drawNetworkGraph();
    drawDashboard();
    window.display();
}

void SimulationUI::drawSparkline(float x, float y, float w, float h) {
    sf::RectangleShape bg(sf::Vector2f(w, h));
    bg.setPosition(sf::Vector2f(x, y));
    bg.setFillColor(sf::Color(15, 17, 20));
    bg.setOutlineThickness(1.f);
    bg.setOutlineColor(sf::Color(45, 50, 60));
    window.draw(bg);

    if (lossHistory.empty()) return;

    float maxLoss = 0.0001f;
    for (float l : lossHistory)
        if (l > maxLoss) maxLoss = l;

    std::vector<sf::Vertex> line;
    for (size_t i = 0; i < lossHistory.size(); ++i) {
        float px = x + (i / 60.f) * w;
        float py = y + h - ((lossHistory[i] / maxLoss) * h);
        sf::Vertex point;
        point.position = sf::Vector2f(px, py);
        point.color    = sf::Color(100, 200, 255);
        line.push_back(point);
    }

    if (line.size() > 1)
        window.draw(line.data(), line.size(), sf::PrimitiveType::LineStrip);
}

void SimulationUI::drawDashboard() {
    float windowHeight = static_cast<float>(window.getSize().y);

    sf::RectangleShape sidebar(sf::Vector2f(360.f, windowHeight));
    sidebar.setFillColor(sf::Color(22, 24, 28));
    window.draw(sidebar);

    uiText.setCharacterSize(22);
    uiText.setFillColor(sf::Color::White);
    uiText.setString("SYSTEM DASHBOARD");
    uiText.setPosition({30.f, 40.f});
    window.draw(uiText);

    // pulsing green when training, solid red when paused
    sf::CircleShape statusLight(6.f);
    statusLight.setPosition({30.f, 135.f});
    if (isTraining) {
        float pulse = (std::sin(currentEpoch * 0.1f) * 63.f) + 192.f;
        statusLight.setFillColor(sf::Color(100, 255, 150, static_cast<std::uint8_t>(pulse)));
    } else {
        statusLight.setFillColor(sf::Color(255, 80, 80));
    }
    window.draw(statusLight);

    uiText.setCharacterSize(16);
    uiText.setFillColor(sf::Color(150, 160, 175));
    std::stringstream controls;
    controls << "STATUS:      " << (isTraining ? "ACTIVE" : "PAUSED") << "\n"
             << "RATE:        " << std::fixed << std::setprecision(4) << learningRate << "\n"
             << "TOGGLE:      [ SPACE ]\n"
             << "TUNE:        [ UP / DOWN ]\n"
             << "EXPORT:      [ S ]";
    uiText.setString(controls.str());
    uiText.setPosition({55.f, 130.f});
    window.draw(uiText);

    // 0.5 threshold for binary output
    int  predLabel = (currentPrediction > 0.5f) ? 1 : 0;
    int  actLabel  = (currentTarget    > 0.5f) ? 1 : 0;
    bool match     = (predLabel == actLabel);

    uiText.setFillColor(match ? sf::Color(120, 230, 140) : sf::Color(100, 200, 255));
    std::stringstream inference;
    inference << "[ LIVE INFERENCE ]\n\n"
              << "Patient ID: #" << currentSampleIndex << "\n"
              << "AI Risk:    " << std::fixed << std::setprecision(1) << currentPrediction * 100.0f << "%\n"
              << "AI Says:    " << (predLabel == 1 ? "DISEASE" : "HEALTHY") << "\n"
              << "Actual:     " << (actLabel  == 1 ? "DISEASE" : "HEALTHY") << "\n"
              << "VERDICT:    " << (match ? "MATCH" : "MISS");
    uiText.setString(inference.str());
    uiText.setPosition({30.f, 300.f});
    window.draw(uiText);

    // letter grade based on loss thresholds
    uiText.setFillColor(sf::Color(150, 160, 175));
    std::string grade = "C";
    if      (lastLoss < 0.005) grade = "A+";
    else if (lastLoss < 0.01)  grade = "A";
    else if (lastLoss < 0.05)  grade = "B";
    else if (lastLoss == 0.0)  grade = "N/A";

    std::stringstream metrics;
    metrics << "[ PERFORMANCE ]\n\n"
            << "EPOCH:      " << currentEpoch << "\n"
            << "CUR LOSS:   " << std::fixed << std::setprecision(6) << lastLoss << "\n"
            << "MODEL GRADE: " << grade << "\n"
            << "SAMPLES:    " << data.size() << "\n\n";

    // copy and drain so we don't destroy the original heap
    std::priority_queue<EpochEntry> pqCopy = bestEpochs;
    std::vector<EpochEntry> ranked;
    while (!pqCopy.empty()) {
        ranked.push_back(pqCopy.top());
        pqCopy.pop();
    }
    metrics << "[ TOP " << TOP_K << " BEST EPOCHS ]\n";
    if (ranked.empty()) {
        metrics << "  (waiting for training...)";
    } else {
        for (int i = static_cast<int>(ranked.size()) - 1; i >= 0; i--)
            metrics << "  #" << ranked[i].second
                    << "  loss " << std::fixed << std::setprecision(5)
                    << ranked[i].first << "\n";
    }
    uiText.setString(metrics.str());
    uiText.setPosition({30.f, 460.f});
    window.draw(uiText);

    uiText.setCharacterSize(12);
    uiText.setString("LOSS HISTORY (LAST 60 EPOCHS)");
    uiText.setPosition({30.f, 690.f});
    window.draw(uiText);

    drawSparkline(30.f, 710.f, 300.f, 80.f);
}

// smooth easing so column reveals don't pop in abruptly
static inline float smoothstep01(float t) {
    if (t < 0.f) t = 0.f;
    if (t > 1.f) t = 1.f;
    return t * t * (3.f - 2.f * t);
}

// linear blend between two colors
static inline sf::Color lerpColor(const sf::Color& a, const sf::Color& b, float t) {
    if (t < 0.f) t = 0.f;
    if (t > 1.f) t = 1.f;
    return sf::Color(
        static_cast<std::uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<std::uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<std::uint8_t>(a.b + (b.b - a.b) * t),
        static_cast<std::uint8_t>(a.a + (b.a - a.a) * t)
    );
}

void SimulationUI::drawNetworkGraph() {
    const auto& all_layers = network.get_layers();
    std::vector<DenseLayer*> denseLayers;
    for (const auto& l : all_layers)
        if (auto* d = dynamic_cast<DenseLayer*>(l.get())) denseLayers.push_back(d);
    if (denseLayers.empty()) return;

    std::vector<std::string> inputLabels = {
        "Age", "Sex", "Chest Pain", "Resting BP",
        "Cholesterol", "Fasting Sugar", "Resting ECG", "Max Heart Rate",
        "Exer. Angina", "ST Depress.", "ST Slope", "Num Vessels",
        "Thal", "BMI", "Smoking", "Diabetes"
    };
    std::vector<std::string> outputLabels = { "Risk" };

    int   numColumns    = static_cast<int>(denseLayers.size()) + 1;
    float windowWidth   = static_cast<float>(window.getSize().x);
    float windowHeight  = static_cast<float>(window.getSize().y);
    float sidebarWidth  = 360.f;
    float graphAreaWidth = windowWidth - sidebarWidth;

    float startX        = sidebarWidth + (graphAreaWidth * 0.22f);
    float endX          = windowWidth  - (graphAreaWidth * 0.15f);
    float layerSpacingX = (endX - startX) / (numColumns - 1);
    float nodeRadius    = 7.f;

    // fcol is a continuous float, how far the wave has swept across columns
    float fcol          = animElapsed / SECONDS_PER_PHASE;
    int   activeColumn  = static_cast<int>(fcol);
    float phaseProgress = fcol - static_cast<float>(activeColumn);

    // how bright each column should be, earlier columns light up first
    auto columnReveal = [&](int col) {
        float t = fcol - static_cast<float>(col);
        if (t <= 0.f) return 0.f;
        if (t >= 1.f) return 1.f;
        return smoothstep01(t);
    };

    // 0.5 threshold for binary classification
    int  predLabel = (currentPrediction > 0.5f) ? 1 : 0;
    int  actLabel  = (currentTarget    > 0.5f) ? 1 : 0;
    bool match     = (predLabel == actLabel);

    // color palette
    const sf::Color colorIdleNode  (20,  28,  36);
    const sf::Color colorActiveCore(60, 220, 175);
    const sf::Color colorRimDim    (70, 100, 130, 180);
    const sf::Color colorRimLit    (150, 240, 220, 255);
    const sf::Color colorGlow      (80, 230, 200, 60);
    const sf::Color colorConnIdle  (110, 120, 135, 18);
    const sf::Color colorConnLit   (170, 235, 215, 200);
    const sf::Color colorOutputMatch(120, 235, 150);
    const sf::Color colorOutputMiss (255, 170, 70);
    const sf::Color colorActual    (80, 180, 255);

    // draw all connections first so nodes render on top
    // weights(in, out): rows = left side nodes, cols = right side nodes
    for (size_t i = 0; i < denseLayers.size(); ++i) {
        DenseLayer* d = denseLayers[i];
        float prevX = startX + (i       * layerSpacingX);
        float currX = startX + ((i + 1) * layerSpacingX);

        int inNodes  = d->weights.rows;
        int outNodes = d->weights.cols;

        float prevSpacingY = (windowHeight * 0.8f) / std::max(1, inNodes);
        float prevStartY   = (windowHeight - (prevSpacingY * (inNodes  - 1))) / 2.f;
        float currSpacingY = (windowHeight * 0.8f) / std::max(1, outNodes);
        float currStartY   = (windowHeight - (currSpacingY * (outNodes - 1))) / 2.f;

        // pulse brightness for this edge group
        float edgeT = fcol - static_cast<float>(i);
        float edgeBrightness = 0.f;
        if (edgeT > 0.f && edgeT < 1.5f) {
            edgeBrightness = std::sin(std::min(edgeT, 1.f) * 3.14159f);
            edgeBrightness = std::max(0.f, edgeBrightness);
        }
        float edgeTail = (fcol > static_cast<float>(i + 1)) ? 0.25f : 0.f;
        float edgeMix  = std::max(edgeBrightness, edgeTail);

        // k = left node, j = right node, weight at data[k][j]
        for (int j = 0; j < outNodes; ++j) {
            for (int k = 0; k < inNodes; ++k) {
                double w    = d->weights.data[k][j];
                double absw = std::abs(w);
                if (absw < 0.05) continue;  // skip near zero weights

                std::uint8_t baseA = static_cast<std::uint8_t>(std::min(255.0, absw * 60.0 + 12.0));
                sf::Color base = colorConnIdle;
                base.a = baseA;

                sf::Color col = lerpColor(base, colorConnLit, edgeMix);

                sf::Vertex line[2];
                line[0].position = {prevX, prevStartY + k * prevSpacingY};
                line[1].position = {currX, currStartY + j * currSpacingY};
                line[0].color = line[1].color = col;
                window.draw(line, 2, sf::PrimitiveType::Lines);
            }
        }
    }

    // draw nodes: col 0 is inputs, col 1+ is outputs of each dense layer
    for (int i = 0; i < numColumns; ++i) {
        int nodes = (i == 0) ? denseLayers[0]->weights.rows
                             : denseLayers[i - 1]->weights.cols;
        float xPos     = startX + (i * layerSpacingX);
        float spacingY = (windowHeight * 0.8f) / std::max(1, nodes);
        float startY   = (windowHeight - (spacingY * (nodes - 1))) / 2.f;

        bool isInputCol  = (i == 0);
        bool isOutputCol = (i == numColumns - 1);
        bool isPulsing   = (i == activeColumn);

        float reveal        = columnReveal(i);
        float pulseEnvelope = isPulsing ? std::sin(phaseProgress * 3.14159f) : 0.f;

        // get cached activations for this column
        const std::vector<float>* acts = nullptr;
        if (i < static_cast<int>(nodeActivations.size())) acts = &nodeActivations[i];

        for (int n = 0; n < nodes; ++n) {
            float yPos = startY + (n * spacingY);

            float activation = (acts && n < static_cast<int>(acts->size())) ? (*acts)[n] : 0.5f;
            if (activation < 0.f) activation = 0.f;
            if (activation > 1.f) activation = 1.f;

            // brightness = reveal * activation strength + pulse boost
            float brightness = reveal * (0.35f + 0.65f * activation) + 0.35f * pulseEnvelope * activation;
            if (brightness > 1.f) brightness = 1.f;

            // outer glow
            if (brightness > 0.05f) {
                float haloR = nodeRadius * (2.4f + 0.6f * pulseEnvelope);
                sf::CircleShape halo(haloR);
                halo.setOrigin({haloR, haloR});
                halo.setPosition({xPos, yPos});
                sf::Color hc = colorGlow;
                hc.a = static_cast<std::uint8_t>(std::min(255.f, hc.a * brightness * 1.6f));
                halo.setFillColor(hc);
                window.draw(halo);
            }

            // filled body
            float bodyR = nodeRadius + 0.6f * pulseEnvelope;
            sf::CircleShape body(bodyR);
            body.setOrigin({bodyR, bodyR});
            body.setPosition({xPos, yPos});

            sf::Color core = lerpColor(colorIdleNode, colorActiveCore, brightness);

            // output node: green if correct, orange if wrong
            if (isOutputCol && reveal > 0.05f) {
                sf::Color target = match ? colorOutputMatch : colorOutputMiss;
                core = lerpColor(colorIdleNode, target, brightness);
            }

            body.setFillColor(core);

            // rim outline
            sf::Color rim = lerpColor(colorRimDim, colorRimLit, brightness);
            if (isOutputCol && reveal > 0.05f) {
                rim = lerpColor(colorRimDim,
                                match ? colorOutputMatch : colorOutputMiss,
                                std::min(1.f, brightness + 0.3f));
                rim.a = 255;
            }
            body.setOutlineThickness(1.5f);
            body.setOutlineColor(rim);
            window.draw(body);

            // feature label on the left
            if (isInputCol && n < static_cast<int>(inputLabels.size())) {
                uiText.setString(inputLabels[n]);
                uiText.setCharacterSize(11);
                uiText.setFillColor(sf::Color(130, 145, 160, static_cast<std::uint8_t>(160 + 80 * reveal)));
                uiText.setPosition({xPos - 115.f, yPos - 7.f});
                window.draw(uiText);
            }

            // output label on the right
            if (isOutputCol) {
                std::string title = (nodes == 1) ? "Heart Disease" :
                                    (n < static_cast<int>(outputLabels.size()) ? outputLabels[n] : "");
                uiText.setString(title);
                uiText.setCharacterSize(13);
                uiText.setFillColor(sf::Color(255, 210, 130));
                uiText.setPosition({xPos + 22.f, yPos - 18.f});
                window.draw(uiText);

                if (nodes == 1) {
                    uiText.setString("0 = Healthy / 1 = Disease");
                    uiText.setCharacterSize(10);
                    uiText.setFillColor(sf::Color(140, 150, 165));
                    uiText.setPosition({xPos + 22.f, yPos - 2.f});
                    window.draw(uiText);

                    if (reveal > 0.5f) {
                        std::string verdictText = (predLabel == 1) ? "DISEASE" : "HEALTHY";
                        sf::Color verdictColor  = match ? colorOutputMatch
                                                        : (predLabel == 1 ? colorOutputMiss : colorActual);
                        uiText.setString("AI: " + verdictText);
                        uiText.setCharacterSize(12);
                        uiText.setFillColor(verdictColor);
                        uiText.setPosition({xPos + 22.f, yPos + 14.f});
                        window.draw(uiText);

                        std::stringstream conf;
                        conf << std::fixed << std::setprecision(1)
                             << (currentPrediction * 100.0f) << "% risk";
                        uiText.setString(conf.str());
                        uiText.setCharacterSize(10);
                        uiText.setFillColor(sf::Color(180, 190, 200));
                        uiText.setPosition({xPos + 22.f, yPos + 30.f});
                        window.draw(uiText);
                    }
                }
            }

            // yellow box around the output node
            if (isOutputCol && reveal > 0.5f && nodes == 1) {
                float boxSize = nodeRadius * 3.4f;
                sf::RectangleShape highlight(sf::Vector2f(boxSize, boxSize));
                highlight.setOrigin({boxSize / 2.f, boxSize / 2.f});
                highlight.setPosition({xPos, yPos});
                highlight.setFillColor(sf::Color::Transparent);
                highlight.setOutlineThickness(1.5f);
                sf::Color frameCol = match ? colorOutputMatch : sf::Color(255, 220, 90);
                frameCol.a = static_cast<std::uint8_t>(180 * std::min(1.f, (reveal - 0.5f) * 2.f));
                highlight.setOutlineColor(frameCol);
                window.draw(highlight);
            }
        }
    }
}
