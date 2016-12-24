(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))


(declare output)

(defn HiddenLayer [w b]
  {:weights w
   :bias b

   :activation (fn [layer input]
                 (let [activate (fn [i w] (reduce + (map * i w)))
                       output   (map #(activate input %) :weights)
                       bias-out (map + output :bias)]
                   (map sigmoid bias-out)))

   :sample-h-given-v (fn [inputs]
                       (mapv (fn [w_i b_i] (binomial 1 (output inputs w_i b_i))) :weights :bias))
  })
     
    ;; helper functions 
    (defn output [inputs weights bias]
      (let [linear-output (reduce + (map * inputs weights))
            linear-output (+ linear-output bias)]
        (sigmoid linear-output)))

    