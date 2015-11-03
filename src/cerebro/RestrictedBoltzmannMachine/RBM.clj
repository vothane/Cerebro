(ns cerebro.RestrictedBoltzmannMachine.RBM
  (:use [cerebro.Utils.utils])
  (:refer-clojure :exclude [* - + == / < <= > >= not= = min max])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

; Boltzmann Machines (BMs) are a particular form of energy-based model which
; contain hidden variables. Restricted Boltzmann Machines further restrict BMs
; to those without visible-visible and hidden-hidden connections.

(defrecord RBM [N
	            n-visible
	            n-hidden
	            weights
	            hbias
	            vbias])

(defn- rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn- uniform [min max] (* (rnd) ( + (- max min) min)))

(defn make-rbm [n nv nh hb vb]
  (->RBM n 
         nv 
         nh 
         (partition nh 
           (take (* nh nv) 
             (repeatly (uniform (* -1 (/ 1 nv)) (/ 1 nv))))) 
         (take hb (repeat 0.0))
         (take vb (repeat 0.0))))
     