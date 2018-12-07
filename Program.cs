using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork {
    internal class Program {
        private static void Main() {
            const double maxIterations = 1e5;

            var architecture = new List<int> {2, 3, 1};
            var brain = new Network(architecture);

            var samples = new List<Sample> {
                new Sample {Input = new List<double> {0, 0}, Output = new List<double> {0}},
                new Sample {Input = new List<double> {0, 1}, Output = new List<double> {1}},
                new Sample {Input = new List<double> {1, 0}, Output = new List<double> {1}},
                new Sample {Input = new List<double> {1, 1}, Output = new List<double> {0}}
            };

            var random = new Random();
            for (var i = 0; i < maxIterations; i++) {
                var index = random.Next(0, samples.Count);
                var sample = samples.ElementAt(index);
                var (input, expected) = (sample.Input, sample.Output);
                var predicted = brain.Predict(input);
                brain.Learn(predicted, expected);
            }

            Console.WriteLine("input\toutput\tpredicted");
            samples.ForEach(sample => {
                var (input, output) = (sample.Input, sample.Output);
                var inputs = string.Join("|", input);
                var expected = string.Join("|", output);
                var predicted = string.Join("|", brain.Predict(input));
                Console.WriteLine($"{inputs}\t{expected}\t{predicted}");
            });
            Console.ReadKey();
        }
    }

    internal class Sample {
        public List<double> Input { get; set; }
        public List<double> Output { get; set; }
    }
}