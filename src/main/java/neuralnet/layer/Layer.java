package neuralnet.layer;

import neuralnet.activationfunction.IActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Layer {
    private INDArray weights;
    private INDArray biases;
    private IActivationFunction activationFunction;

    // Iteration instance variables.
    private INDArray z;
    private INDArray activation;
    private INDArray activationDerivative;

    public static Builder Builder() {
        return new Builder();
    }

    protected Layer(INDArray weights, INDArray biases, IActivationFunction activationFunction) {
        this.weights = weights;
        this.biases = biases;
        this.activationFunction = activationFunction;
    }

    public void activate(INDArray input) {
        z = calculateZ(input);
        activation = calculateActivation(z);
        activationDerivative = getActivationDerivative(z);
    }

    public INDArray getZ() {
        return z;
    }

    public INDArray getActivation() {
        return activation;
    }

    public INDArray getActivationDerivative() {
        return activationDerivative;
    }

    public INDArray getErrorGradient(INDArray error) {
        return weights.transpose().mmul(error);
    }

    public void updateWeights(INDArray gradients) {
        weights = weights.add(gradients);
    }

    public void updateBiases(INDArray gradients) {
        biases = biases.add(gradients);
    }

    private INDArray calculateZ(INDArray input) {
        return weights.mmul(input).add(biases);
    }

    private INDArray calculateActivation(INDArray z) {
        return activationFunction.output(z);
    }

    private INDArray getActivationDerivative(INDArray z) {
        return activationFunction.derivative(z);
    }
}
