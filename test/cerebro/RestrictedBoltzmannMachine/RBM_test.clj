(ns cerebro.RestrictedBoltzmannMachine.RBM_test
  (:require [clojure.test :refer :all]
            [cerebro.RestrictedBoltzmannMachine.RBM :refer :all]))
 
(deftest prop-up-test
  (testing "RBM propagate up"
    (let [v [0 1 1]
          W [-3.85666 1.17140 3.71995]   
          b -0.93577]
      (is (= (RBM-propup v W b) 0.9812121811403206)))))

(deftest prop-down-test
  (testing "RBM propagate down"
    (let [h [0 1 1]
          W [[2.26799 -0.03222 -2.32391] [0.71964 0.35312 -0.69871] [-3.85666 1.17140 3.71995]]
          b -0.13333
          i 2]
      (is (= (RBM-propdown h W b i) 0.9472455388518453)))))
