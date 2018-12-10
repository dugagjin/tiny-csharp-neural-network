using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork {
    /// <summary>
    ///     Represents a simple <see cref="Neuron" /> that can be used as standalone,
    ///     in a <see cref="Layer" /> or in a <see cref="Network" />.
    /// </summary>
    public class Neuron {
        private static readonly Random Random = new Random();
        private readonly double _learningRate;
        private double _error;
        private IEnumerable<double> _input;
        private double _output;
        private IEnumerable<double> _weights;

        /// <summary>
        ///     Create an instance of a single <see cref="Neuron" />.
        /// </summary>
        /// <param name="inputs">Number of input data.</param>
        /// <param name="learningRate">Rate of 1 is normal rate, 0 is no rate.</param>
        public Neuron(int inputs, double learningRate) {
            _error = 0;
            _output = 0;
            _learningRate = learningRate;
            _weights = Enumerable.Range(0, inputs).Select(_ => Random.NextDouble() - 0.5);
        }

        /// <summary>
        ///     Predict the output according to the input.
        ///     Element-wise multiplication followed by a sum and finally a sigmoid activation function.
        /// </summary>
        /// <param name="input">Sequence of input values.</param>
        /// <returns>Predicted output.</returns>
        public double Predict(IEnumerable<double> input) {
            _input = input.ToList();
            _output = Multiply(_input, _weights).Sum();
            return Sigmoid(_output);
        }

        /// <summary>
        ///     Performs the inverse of prediction. Each weight, except bias, is multiplied with the error.
        /// </summary>
        /// <param name="error">Difference between predicted and expected output.</param>
        /// <returns>Weights, except bias, scaled with the error.</returns>
        public IEnumerable<double> BackPropagation(double error) {
            _error = error;
            return _weights.Skip(1).Select(weight => weight * error);
        }

        /// <summary>
        ///     Update the weights of the neuron according to the input, output, error and learning rate.
        /// </summary>
        public void Update() {
            var deltas = _input.Select(input => input * SigmoidDerivative(_output) * _error * _learningRate);
            _weights = Subtract(_weights, deltas).ToList();
        }

        private static double Sigmoid(double z) => 1 / (1 + Math.Exp(z * -1));

        private static double SigmoidDerivative(double z) => Sigmoid(z) * (1 - Sigmoid(z));

        private static IEnumerable<double> Multiply(IEnumerable<double> arr1, IEnumerable<double> arr2) =>
            arr1.Select((e, i) => e * arr2.ElementAt(i));

        private static IEnumerable<double> Subtract(IEnumerable<double> arr1, IEnumerable<double> arr2) =>
            arr1.Select((e, i) => e - arr2.ElementAt(i));
    }
}