package neuralnet;

import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedList;
import java.util.List;

public class NeuralNet {
    private List<Layer> layers;
    private double learningRate;

    public static final Builder BUILDER = new Builder();

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
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.activate(layerInput);
            INDArray output = layer.getActivation();

            activations.add(output);
            layerInput = output;
        }

        return activations;
    }

    public List<INDArray> backpropagateError(INDArray error) {
        List<INDArray> gradients = new LinkedList<INDArray>();
        gradients.add(error);

        INDArray delta = error;
        for (int i = layers.size() - 1; i >= 2; i--) {
            Layer layer = layers.get(i);

            delta = layer.getErrorGradient(delta);
            gradients.add(layers.size() - i - 1, delta);
        }

        return gradients;
    }

    public void train(INDArray data, INDArray expected) {
        feedForward(data);

        Layer outputLayer = layers.get(layers.size() - 1);
        INDArray output = outputLayer.getZ();
        INDArray activationDerivative = outputLayer.getActivationDerivative();
        INDArray error = expected.sub(output).mul(activationDerivative);

        List<INDArray> errorGradients = backpropagateError(error);

        for (int i = layers.size() - 1; i >=1 ; i--) {
            Layer layer = layers.get(i);
            Layer previousLayer = layers.get(i - 1);

            INDArray errorGradient = errorGradients.get(i - 1);
            INDArray activation = previousLayer.getActivation();

            // Use data as the activation for input layer.
            if (i == 1) {
                activation = data;
            }

            INDArray weightDelta = errorGradient.mmul(activation.transpose()).mul(learningRate);

            layer.updateWeights(weightDelta);
            layer.updateBiases(errorGradient);
        }
    }
}
