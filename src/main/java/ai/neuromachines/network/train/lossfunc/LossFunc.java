package ai.neuromachines.network.train.lossfunc;

import ai.neuromachines.network.function.ActivationFunc;

/**
 * Loss function for backpropagation error calculation
 */
public interface LossFunc {

    static LossFunc crossEntropy() {
        return CrossEntropyLossFunc.of();
    }

    static LossFunc leastSquares() {
        return LeastSquaresLossFunc.of();
    }

    /**
     * Checks for correctness of network expected output values for this loss function
     */
    void verifyExpectedOutput(float[] expectedOutput);

    /**
     * Calculates δ (delta) for last layer's node
     *
     * @see <a href="https://en.wikipedia.org/wiki/Backpropagation">Backpropagation</a>
     */
    float calculateDelta(float input, float output, float expectedOutput, ActivationFunc func);
}
