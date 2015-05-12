(ns cerebro.Utils.utils
  (:use clojure.core.matrix)
  (:require [clojure.core.matrix.operators :as M]))

(defn dp [m1 m2]
  (inner-product m1 m2))  