package ai.neuromachines.network.train.lossfunc;

import ai.neuromachines.network.function.ActivationFunc;

class LeastSquaresLossFunc implements LossFunc {
    static final LeastSquaresLossFunc INSTANCE = new LeastSquaresLossFunc();

    static LeastSquaresLossFunc of() {
        return INSTANCE;
    }

    @Override
    public void verifyExpectedOutput(float[] expectedOutput) {
        // any value is acceptable
    }

    @Override
    public float calculateDelta(float input, float output, float expectedOutput, ActivationFunc func) {
        float error = output - expectedOutput;
        float derivative = func.derivative().apply(input);
        return error * derivative;
    }
}
