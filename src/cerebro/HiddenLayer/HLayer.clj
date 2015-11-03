(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils])
  (:refer-clojure :exclude [* - + == / < <= > >= not= = min max])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

(defrecord HLayer [N
	               n-inputs
	               n-outputs
	               weights
	               bias])

(defn- rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn- uniform [min max] (* (rnd) ( + (- max min) min)))

(defn make-hidden-layer [n n-in n-out]
  (->HLayer n 
            n-in
            n-out 
            (partition n-in 
              (take (* n-in n-out) 
                (repeatly (uniform (* -1 (/ 1 n-in)) (/ 1 n-in))))) 
            (take b (repeat 0.0))))
     