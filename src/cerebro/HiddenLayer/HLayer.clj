(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

(defn output [input weights bias]
	(let [linear-output (reduce + (map * weights bias))
        linear-output (+ linear-ouput bias)]
    (sigmoid linear-output)))

(defn sample-h-given-v [hidden-layer biases input]
  (map #(binomial 1 (output hidden-layer input %1 %2)) hidden-layer biases))