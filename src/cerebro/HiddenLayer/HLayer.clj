(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

;; API functions
(defn activation [sigmoid input] ((:activation sigmoid) input))

(declare output)

(defn HiddenLayer [weights bias]
  {:activation (fn [input]
                 (let [activate (fn [i w] (reduce + (map * i w)))
                       output   (map #(activate input %) weights)
                       bias-out (map + output bias)]
                   (map sigmoid bias-out)))

   :sample-h-given-v (fn [inputs]
                       (mapv (fn [w_i b_i] (binomial 1 (output inputs w_i b_i))) weights bias))
  })
     
    ;; helper functions 
    (defn output [inputs weights bias]
      (let [linear-output (reduce + (map * inputs weights))
            linear-output (+ linear-output bias)]
        (sigmoid linear-output)))

    