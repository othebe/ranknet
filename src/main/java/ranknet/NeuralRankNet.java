package ranknet;

import com.sun.tools.javac.util.Pair;
import neuralnet.layer.Layer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class NeuralRankNet {
    protected List<Layer> layers;
    protected double learningRate;

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

    public void train(INDArray inputI, INDArray inputJ, INDArray expected) {
        Layer outputLayer = layers.get(layers.size() - 1);

        Pair<List<INDArray>, List<INDArray>> pairI = getActivationZPair(inputI);
        Pair<List<INDArray>, List<INDArray>> pairJ = getActivationZPair(inputJ);

        List<INDArray> activationsI = pairI.fst;
        List<INDArray> zI = pairI.snd;

        List<INDArray> activationsJ = pairJ.fst;
        List<INDArray> zJ = pairJ.snd;

        INDArray oI = activationsI.get(activationsI.size() - 1);
        INDArray oJ = activationsJ.get(activationsJ.size() - 1);
        INDArray oIJ = oI.sub(oJ);
        INDArray pIJ = getProbability(oIJ);

        INDArray deltaC = getOutputCostDerivative(pIJ, expected);
        INDArray outputErrorGradientI = deltaC.mul(outputLayer.calculateActivationDerivative(oI));
        INDArray outputErrorGradientJ = deltaC.mul(outputLayer.calculateActivationDerivative(oJ));

        List<INDArray> errorGradientsI = backpropagateError(outputErrorGradientI, activationsI);
        List<INDArray> errorGradientsJ = backpropagateError(outputErrorGradientJ, activationsJ);

        updateParams(
                new Pair<List<INDArray>, List<INDArray>>(errorGradientsI, errorGradientsJ),
                new Pair<List<INDArray>, List<INDArray>>(activationsI, activationsJ));
    }

    private List<INDArray> backpropagateError(INDArray outputErrorGradient, List<INDArray> activations) {
        Stack<INDArray> reverseGradients = new Stack<INDArray>();
        INDArray errorGradient = outputErrorGradient;
        reverseGradients.add(errorGradient);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer layer = layers.get(i);
            Layer forwardLayer = layers.get(i + 1);
            INDArray activationDerivative = layer.calculateActivationDerivative(activations.get(i + 1));
            errorGradient = forwardLayer.getErrorGradient(errorGradient, activationDerivative);
            reverseGradients.add(errorGradient);
        }

        List<INDArray> errorGradients = new LinkedList<INDArray>();
        while (!reverseGradients.isEmpty()) {
            errorGradients.add(reverseGradients.pop());
        }

        return errorGradients;
    }

    private void updateParams(Pair<List<INDArray>, List<INDArray>> errorGradients, Pair<List<INDArray>, List<INDArray>> activations) {
        List<INDArray> errorGradientsI = errorGradients.fst;
        List<INDArray> errorGradientsJ = errorGradients.snd;

        List<INDArray> activationsI = activations.fst;
        List<INDArray> activationsJ = activations.snd;

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            INDArray errorGradientI = errorGradientsI.get(i);
            INDArray errorGradientJ = errorGradientsJ.get(i);

            INDArray activationI = activationsI.get(i);
            INDArray activationJ = activationsJ.get(i);

            INDArray weightDeltaI = errorGradientI.transpose().mmul(activationI).mul(learningRate);
            INDArray weightDeltaJ = errorGradientJ.transpose().mmul(activationJ).mul(learningRate);

            layer.updateWeights(weightDeltaI.sub(weightDeltaJ));
            layer.updateBiases(errorGradientI.sub(errorGradientJ));
        }
    }

    private static INDArray getProbability(INDArray data) {
        return Transforms.sigmoid(data);
    }

    private static INDArray getOutputCostDerivative(INDArray output, INDArray expected) {
        // Naive MSE.
        // return expected.sub(output);

        // XE
        ILossFunction lf = new LossMSE();
        return lf.computeGradient(expected, output, new ActivationIdentity(), null);
    }

    private Pair<List<INDArray>, List<INDArray>> getActivationZPair(INDArray input) {
        List<INDArray> activations = feedForward(input);
        List<INDArray> zs = new LinkedList<INDArray>();
        zs.add(input);
        for (Layer layer : layers) {
            zs.add(layer.getZ());
        }

        return new Pair<List<INDArray>, List<INDArray>>(activations, zs);
    }
}
