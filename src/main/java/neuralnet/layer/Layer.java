package neuralnet.layer;

import neuralnet.activationfunction.IActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.List;

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
        return error.mmul(weights);
    }

    public void updateWeights(INDArray gradients) {
        List<INDArray> rows = new LinkedList<INDArray>();
        int stride = activation.rows() * weights.rows();

        for (int row = 0; row < gradients.rows(); row = row + stride) {
            int[] extractRows = new int[stride];
            for (int i = 0; i < stride; i++) {
                extractRows[i] = (row * stride) + i;
            }
            rows.add(gradients.getRows(extractRows));
        }
        weights = weights.add(Nd4j.averageAndPropagate(rows));
    }

    public void updateBiases(INDArray gradients) {
        List<INDArray> rows = new LinkedList<INDArray>();
        int stride = activation.rows();

        for (int row = 0; row < gradients.rows(); row = row + stride) {
            int[] extractRows = new int[stride];
            for (int i = 0; i < stride; i++) {
                extractRows[i] = (row * stride) + i;
            }
            rows.add(gradients.getRows(extractRows));
        }
        biases = biases.add(Nd4j.averageAndPropagate(rows));
    }

    private INDArray calculateZ(INDArray input) {
        INDArray biasMatrix = biases;
        for (int i = 1; i < input.rows(); i++) {
            biasMatrix = Nd4j.hstack(biasMatrix, biases);
        }
        return input.mmul(weights.transpose()).add(biasMatrix);
    }

    private INDArray calculateActivation(INDArray z) {
        return activationFunction.output(z);
    }

    private INDArray getActivationDerivative(INDArray z) {
        return activationFunction.derivative(z);
    }
}
