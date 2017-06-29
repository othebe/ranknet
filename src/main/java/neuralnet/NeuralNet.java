package neuralnet;

import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class NeuralNet {
    private List<Layer> layers;
    private double learningRate;

    public static Builder Builder() {
        return new Builder();
    }

    protected void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    protected void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public List<INDArray> feedForward(INDArray input) {
        List<INDArray> activations = new LinkedList<INDArray>();
        activations.add(input);

        INDArray layerInput = input;
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.activate(layerInput);
            INDArray output = layer.getActivation();

            activations.add(output);
            layerInput = output;
        }

        return activations;
    }

    public List<INDArray> backpropagateError(INDArray error) {
        Stack<INDArray> reverseGradients = new Stack<INDArray>();
        List<INDArray> gradients = new LinkedList<INDArray>();

        reverseGradients.push(error);

        INDArray delta = error;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);

            delta = layer.getErrorGradient(delta);
            reverseGradients.push(delta);
        }

        while (!reverseGradients.isEmpty()) {
            gradients.add(reverseGradients.pop());
        }

        return gradients;
    }

    public void backpropagateAndUpdate(INDArray input, INDArray error) {
        List<INDArray> errorGradients = backpropagateError(error);

        for (int i = layers.size() - 1; i >=0 ; i--) {
            Layer layer = layers.get(i);
            INDArray errorGradient = errorGradients.get(i + 1);

            INDArray activation = (i == 0) ? input : layers.get(i - 1).getActivation();

            INDArray weightDelta = errorGradient.transpose().mmul(activation).mul(learningRate);

            layer.updateWeights(weightDelta);
            layer.updateBiases(errorGradient);
        }
    }

    public void train(INDArray data, INDArray expected) {
        feedForward(data);

        Layer outputLayer = layers.get(layers.size() - 1);
        INDArray output = outputLayer.getZ();
        INDArray activationDerivative = outputLayer.getActivationDerivative();
        INDArray error = expected.sub(output).mul(activationDerivative);

        backpropagateAndUpdate(data, error);
    }
}
