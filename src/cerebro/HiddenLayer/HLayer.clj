(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

;; API functions
(defn activation [sigmoid input] ((:activation sigmoid) input))
(defn sigmoid-sample-h-given-v [sigmoid input] ((:sample-h-given-v sigmoid) input))

(declare output)

(defn HiddenLayer [weights bias]
  {:activation (fn [input]
                 (let [activate (fn [i w] (reduce + (map * i w)))
                       output   (map #(activate input %) weights)
                       bias-out (map + output bias)]
                   (map sigmoid bias-out)))

   :sample-h-given-v (fn [inputs]
                       (mapv 
                         (fn [input-vector weight-vector bias-scalar] 
                           (binomial 1 (output input-vector weight-vector bias-scalar))) 
                         inputs weights bias))
  })
     
    ;; helper functions 
    (defn output [input-vector weight-vector bias-scalar]
      (let [linear-output (-> (dot-product input-vector weight-vector) (+ bias-scalar))]
        (sigmoid linear-output)))

    