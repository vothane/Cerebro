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

(deftest sample-h-given-v-test
  (testing "RBM sample h given v"
    (let [hbias  [0.028221124673994354 -0.107446264843636060 -0.9357719706191907]
          v  [0 1 1]
          W [[ 2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
             [ 0.7196361212711985  0.353115438019772200 -0.6987101836277690] 
             [-3.8566569510903440  1.171395266717829100  3.7199477886071834]]
          
          means [0.08883754884996532 0.38863798230998053 0.981212016787623]
          s_h|v (RBM-sample-h-given-v hbias W v)]
      (is (= (:means s_h|v) means))
      (is (every? #{0 1} (:samples s_h|v))))))

(deftest sample-v-given-h-test
  (testing "RBM sample v given h"
    (let [vbias [0.39999999999999997 1.2166666666666672 -0.13333333333333333]  
          h  [0 1 1]
          W [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
             [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
             [-3.856656951090344 1.1713952667178291 3.7199477886071834]]
          
          means [0.06082386389198515 0.9394131427398393 0.9472452525970866]
          s-v|h (RBM-sample-v-given-h vbias W h)]
      (is (= (:means s-v|h) means))
      (is (every? #{0 1} (:samples s-v|h))))))

(deftest gibbs-hvh-test
  (testing "RBM Gibbs sampling step: hidden -> visible -> hidden"
    (let [vbias [0.39999999999999997 1.2166666666666672 -0.13333333333333333]  
          hbias [0.028221124673994354 -0.107446264843636060 -0.9357719706191907]
          h  [0 1 1]
          W [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
             [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
             [-3.856656951090344 1.1713952667178291 3.7199477886071834]]
          
          means-h|v [0.08883754884996532 0.38863798230998053 0.981212016787623]
          means-v|h [0.06082386389198515 0.9394131427398393 0.9472452525970866]
          gibbs-hvh (RBM-gibbs-hvh hbias vbias W h)]
      (is (= (:means (:v|h gibbs-hvh)) means-v|h))
      (is (every? #{0 1} (:samples (:v|h gibbs-hvh)))) 
      (is (= (:means (:h|v gibbs-hvh)) means-h|v))
      (is (every? #{0 1} (:samples (:h|v gibbs-hvh)))))))

