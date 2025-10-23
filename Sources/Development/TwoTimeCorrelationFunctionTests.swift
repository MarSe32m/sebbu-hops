//
//  TwoTimeCorrelationFunctionTests.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 20.10.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

func testSingleParticleLinearHOPSTwoTimeCorrelationFunction(trajectories: Int) {
    let A = 0.027
    let omegaC = 1.447
    let endTime = 50.0
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, Complex(0.1), Complex(0.1), Complex(0.1)], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let noises = zGenerator.generateParallel(count: trajectories)
    
    let O: Matrix<Complex<Double>> = .init(elements: [.one, .zero, .zero, -.one], rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    var tSpace: [Double] = []
    var expectationValue: [Complex<Double>] = []
    noises.parallelMap { z in
        let (_tSpace, trajectory) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z)
        let expVal = trajectory.map { $0.inner($0, metric: O) }
        return (_tSpace, expVal)
    }.forEach { _tSpace, expVal in
        if tSpace.isEmpty {
            tSpace = _tSpace
            expectationValue = expVal.map { $0 / Double(trajectories) }
        } else {
            for i in expectationValue.indices {
                expectationValue[i] += expVal[i] / Double(trajectories)
            }
        }
    }
    
    let O1: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let O2: Matrix<Complex<Double>> = .identity(rows: 2) - O1
    
    let sSpace = [Double].linearSpace(0, endTime, 20)
    for s in sSpace {
        var tauSpace: [Double] = []
        var expO: [Complex<Double>] = []
        noises.parallelMap { z in
            let (_tauSpace1, bra1, ket1, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z)
            let (_, bra2, ket2, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z)
            let expO1 = zip(bra1, ket1).map { bra, ket in
                return bra.inner(ket, metric: O)
            }
            let expO2 = zip(bra2, ket2).map { bra, ket in
                return bra.inner(ket, metric: O)
            }
            let expO = zip(_tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
                if tau < s {
                    return 0.5 * (O1O2.0 + O1O2.1)
                } else {
                    return O1O2.0 + O1O2.1
                }
            }
            return (_tauSpace1, expO)
        }.forEach { _tauSpace, expVal in
            if tauSpace.isEmpty {
                tauSpace = _tauSpace
                expO = expVal.map { $0 / Double(trajectories) }
            } else {
                for i in expO.indices {
                    expO[i] += expVal[i] / Double(trajectories)
                }
            }
        }
        
        
//        let (tauSpace1, bra1, ket1, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z)
//        let (tauSpace2, bra2, ket2, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z)
//        
//        plt.figure()
//        plt.plot(x: tSpace, y: trajectory.map { $0[0].real }, label: "Re P[0]")
//        plt.plot(x: tSpace, y: trajectory.map { $0[0].imaginary }, label: "Im P[0]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[0].real }, label: "Re P1[0]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[0].imaginary }, label: "Im P1[0]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[0].real }, label: "Re P2[0]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[0].imaginary }, label: "Im P2[0]")
//        plt.legend()
//        plt.show()
//        plt.close()
//        
//        plt.figure()
//        plt.plot(x: tSpace, y: trajectory.map { $0[1].real }, label: "Re P[1]")
//        plt.plot(x: tSpace, y: trajectory.map { $0[1].imaginary }, label: "Im P[1]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[1].real }, label: "Re P1[1]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[1].imaginary }, label: "Im P1[1]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[1].real }, label: "Re P2[1]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[1].imaginary }, label: "Im P2[1]")
//        plt.legend()
//        plt.show()
//        plt.close()
//        
//        let expO1 = zip(bra1, ket1).map { $0.inner($1, metric: O) }
//        let expO2 = zip(bra2, ket2).map { $0.inner($1, metric: O) }
//        let expO = zip(tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
//            if tau < s {
//                return 0.5 * (O1O2.0 + O1O2.1)
//            } else {
//                return O1O2.0 + O1O2.1
//            }
//        }
        plt.figure()
        plt.plot(x: tSpace, y: expectationValue.real, label: "Re <O>")
        plt.plot(x: tSpace, y: expectationValue.imaginary, label: "Im <O>")
        plt.plot(x: tauSpace, y: expO.real, label: "Re (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.plot(x: tauSpace, y: expO.imaginary, label: "Im (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.title("Linear: s=\(s)")
        plt.legend()
        plt.show()
        plt.close()
    }
}

func testSingleParticleNonLinearHOPSTwoTimeCorrelationFunction(trajectories: Int) {
    let A = 0.027
    let omegaC = 1.447
    let endTime = 750.0
    let gammaR = 0.0175
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, Complex(0.1), Complex(0.1), Complex(0.1)], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let noises = zGenerator.generateParallel(count: trajectories)
    
    let whiteNoiseGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaR / 2, start: 0.0, end: endTime, step: 0.01)
    let whiteNoises = whiteNoiseGenerator.generateParallel(count: trajectories)
    
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = sigmaMinus.conjugateTranspose
    let spsm = -gammaR / 2 * sigmaMinus.conjugateTranspose.dot(sigmaMinus)
    
    let O: Matrix<Complex<Double>> = .init(elements: [.one, .zero, .zero, -.one], rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    var tSpace: [Double] = []
    var expectationValue: [Complex<Double>] = []
    zip(noises, whiteNoises).parallelMap { [H] (z, w) in
        var _O: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
        let customOperator: (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
            let sigmaPlusExp = state.inner(state, metric: sigmaPlus) / state.normSquared
            _O.copyElements(from: spsm)
            _O.add(sigmaMinus, multiplied: gammaR * sigmaPlusExp)
            return _O
        }
        let (_tSpace, trajectory) = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, customOperators: [customOperator], stepSize: 0.1)
        let expVal = trajectory.map { $0.inner($0, metric: O) / $0.normSquared }
        return (_tSpace, expVal)
    }.forEach { _tSpace, expVal in
        if tSpace.isEmpty {
            tSpace = _tSpace
            expectationValue = expVal.map { $0 / Double(trajectories) }
        } else {
            for i in expectationValue.indices {
                expectationValue[i] += expVal[i] / Double(trajectories)
            }
        }
    }
    
    
    let O1: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let O2: Matrix<Complex<Double>> = .identity(rows: 2) - O1
    
    let sSpace = [Double].linearSpace(0, endTime, 20)
    for s in sSpace {
        var tauSpace: [Double] = []
        var expO: [Complex<Double>] = []
        zip(noises, whiteNoises).parallelMap { [H] z, w in
            var _O1: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
            let customOperatorBra: (Double, Vector<Complex<Double>>, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, bra, ket in
                let sigmaPlusExp = bra.inner(bra, metric: sigmaPlus) / bra.normSquared
                _O1.copyElements(from: spsm)
                _O1.add(sigmaMinus, multiplied: gammaR * sigmaPlusExp)
                return _O1
            }
            var _O2: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
            let customOperatorKet: (Double, Vector<Complex<Double>>, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, bra, ket in
                let sigmaPlusExp = bra.inner(bra, metric: sigmaPlus) / bra.normSquared
                _O2.copyElements(from: spsm)
                _O2.add(sigmaMinus, multiplied: gammaR * sigmaPlusExp)
                return _O2
            }
            let (_tauSpace1, bra1, ket1, _, normalization1) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, braCustomOperators: [customOperatorBra], ketCustomOperators: [customOperatorKet], stepSize: 0.1)
            let (_, bra2, ket2, _, normalization2) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, braCustomOperators: [customOperatorBra], ketCustomOperators: [customOperatorKet], stepSize: 0.1)
            let expO1 = zip(normalization1, zip(bra1, ket1)).map { normSquared, braKet in
                let bra = braKet.0
                let ket = braKet.1
                return bra.inner(ket, metric: O) / normSquared
            }
            let expO2 = zip(normalization2, zip(bra2, ket2)).map { normSquared, braKet in
                let bra = braKet.0
                let ket = braKet.1
                return bra.inner(ket, metric: O) / normSquared
            }
            let expO = zip(_tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
                if tau < s {
                    return 0.5 * (O1O2.0 + O1O2.1)
                } else {
                    return O1O2.0 + O1O2.1
                }
            }
            return (_tauSpace1, expO)
        }.forEach { _tauSpace, expVal in
            if tauSpace.isEmpty {
                tauSpace = _tauSpace
                expO = expVal.map { $0 / Double(trajectories) }
            } else {
                for i in expO.indices {
                    expO[i] += expVal[i] / Double(trajectories)
                }
            }
        }
        
//        let (tauSpace1, bra1, ket1, _, normalizationFactor1) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z)
//        let (tauSpace2, bra2, ket2, _, normalizationFactor2) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z)
        
//        plt.figure()
//        plt.plot(x: tSpace, y: trajectory.map { $0[0].real }, label: "Re P[0]")
//        plt.plot(x: tSpace, y: trajectory.map { $0[0].imaginary }, label: "Im P[0]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[0].real }, label: "Re P1[0]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[0].imaginary }, label: "Im P1[0]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[0].real }, label: "Re P2[0]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[0].imaginary }, label: "Im P2[0]")
//        plt.legend()
//        plt.show()
//        plt.close()
        
//        plt.figure()
//        plt.plot(x: tSpace, y: trajectory.map { $0[1].real }, label: "Re P[1]")
//        plt.plot(x: tSpace, y: trajectory.map { $0[1].imaginary }, label: "Im P[1]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[1].real }, label: "Re P1[1]")
//        plt.plot(x: tauSpace1, y: ket1.map { $0[1].imaginary }, label: "Im P1[1]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[1].real }, label: "Re P2[1]")
//        plt.plot(x: tauSpace2, y: ket2.map { $0[1].imaginary }, label: "Im P2[1]")
//        plt.legend()
//        plt.show()
//        plt.close()
        
//        let expO1 = zip(normalizationFactor1, zip(bra1, ket1)).map { normSquared, braKet in
//            let bra = braKet.0
//            let ket = braKet.1
//            return bra.inner(ket, metric: O) / normSquared
//        }
//        let expO2 = zip(normalizationFactor2, zip(bra2, ket2)).map { normSquared, braKet in
//            let bra = braKet.0
//            let ket = braKet.1
//            return bra.inner(ket, metric: O) / normSquared
//        }
//        let expO = zip(tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
//            if tau < s {
//                return 0.5 * (O1O2.0 + O1O2.1)
//            } else {
//                return O1O2.0 + O1O2.1
//            }
//        }
        plt.figure()
        plt.plot(x: tSpace, y: expectationValue.real, label: "Re <O>")
        plt.plot(x: tSpace, y: expectationValue.imaginary, label: "Im <O>")
        plt.plot(x: tauSpace, y: expO.real, label: "Re (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.plot(x: tauSpace, y: expO.imaginary, label: "Im (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.title("Non-linear s=\(s)")
        plt.legend()
        plt.show()
        plt.close()
    }
}
