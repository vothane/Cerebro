(ns cerebro.Utils.utils_test
  (:require [clojure.test :refer :all]
            [cerebro.Utils.utils :refer :all]))

(deftest matrix-elwise-ops-test
  (testing "element-wise ops in a matrix"
    (let [matrix [[0 1]
	              [2 3]]  
          f (fn [m i j] 9)
          -matrix (reduce (fn [m [i j]] (put m i j f)) 
          	        matrix 
          	        (for [i (range-rows matrix) j (range-cols matrix)] [i j]))]
      (is (= -matrix [[9 9] [9 9]])))))
