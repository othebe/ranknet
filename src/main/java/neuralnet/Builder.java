package neuralnet;

import neuralnet.layer.Layer;

import java.util.LinkedList;
import java.util.List;

public class Builder {
    private List<Layer> layers;
    private double learningRate = 1.0f;

    public Builder() {
        this.layers = new LinkedList<Layer>();
    }

    public Builder addLayer(Layer layer) {
        layers.add(layer);

        return this;
    }

    public Builder setLearningRate(double rate) {
        this.learningRate = rate;

        return this;
    }

    public NeuralNet build() {
        NeuralNet neuralNet = new NeuralNet();
        neuralNet.setLayers(layers);
        neuralNet.setLearningRate(learningRate);

        return neuralNet;
    }
}
