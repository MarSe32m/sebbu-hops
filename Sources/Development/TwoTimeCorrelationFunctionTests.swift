//
//  TwoTimeCorrelationFunctionTests.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 20.10.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

func testSingleParticleLinearHOPSTwoTimeCorrelationFunction() {
    let A = 0.027
    let omegaC = 1.447
    let endTime = 10.0
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(0.1)], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    
    let z = GaussianFFTNoiseProcess(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    
    let O: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    let (tSpace, trajectory) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z)
    let expectationValue = trajectory.map { $0.inner($0, metric: O) }
    
    let O1: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let O2: Matrix<Complex<Double>> = .identity(rows: 2) - O1
    
    let sSpace = [Double].linearSpace(0, endTime, 20)
    for s in sSpace {
        let (tauSpace1, bra1, ket1, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z)
        let (tauSpace2, bra2, ket2, _) = hierarchy.solveLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z)
        
        plt.figure()
        plt.plot(x: tSpace, y: trajectory.map { $0[0].real }, label: "Re P[0]")
        plt.plot(x: tSpace, y: trajectory.map { $0[0].imaginary }, label: "Im P[0]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[0].real }, label: "Re P1[0]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[0].imaginary }, label: "Im P1[0]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[0].real }, label: "Re P2[0]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[0].imaginary }, label: "Im P2[0]")
        plt.legend()
        plt.show()
        plt.close()
        
        plt.figure()
        plt.plot(x: tSpace, y: trajectory.map { $0[1].real }, label: "Re P[1]")
        plt.plot(x: tSpace, y: trajectory.map { $0[1].imaginary }, label: "Im P[1]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[1].real }, label: "Re P1[1]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[1].imaginary }, label: "Im P1[1]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[1].real }, label: "Re P2[1]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[1].imaginary }, label: "Im P2[1]")
        plt.legend()
        plt.show()
        plt.close()
        
        let expO1 = zip(bra1, ket1).map { $0.inner($1, metric: O) }
        let expO2 = zip(bra2, ket2).map { $0.inner($1, metric: O) }
        let expO = zip(tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
            if tau < s {
                return 0.5 * (O1O2.0 + O1O2.1)
            } else {
                return O1O2.0 + O1O2.1
            }
        }
        plt.figure()
        plt.plot(x: tSpace, y: expectationValue.real, label: "Re <O>")
        plt.plot(x: tSpace, y: expectationValue.imaginary, label: "Im <O>")
        plt.plot(x: tauSpace1, y: expO.real, label: "Re (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.plot(x: tauSpace1, y: expO.imaginary, label: "Im (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.title("s=\(s)")
        plt.legend()
        plt.show()
        plt.close()
    }
}

func testSingleParticleNonLinearHOPSTwoTimeCorrelationFunction() {
    let A = 0.027
    let omegaC = 1.447
    let endTime = 10.0
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(0.1)], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    
    let z = GaussianFFTNoiseProcess(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    
    let O: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    let (tSpace, trajectory) = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z)
    let expectationValue = trajectory.map { $0.inner($0, metric: O) / $0.normSquared }
    
    let O1: Matrix<Complex<Double>> = .init(elements: Complex<Double>.random(count: 4, in: -1...1), rows: 2, columns: 2)
    let O2: Matrix<Complex<Double>> = .identity(rows: 2) - O1
    
    let sSpace = [Double].linearSpace(0, endTime, 20)
    for s in sSpace {
        let (tauSpace1, bra1, ket1, _, normalizationFactor1) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O1, initialState: initialState, H: H, z: z)
        let (tauSpace2, bra2, ket2, _, normalizationFactor2) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: endTime, A: O, s: s, B: O2, initialState: initialState, H: H, z: z)
        
        plt.figure()
        plt.plot(x: tSpace, y: trajectory.map { $0[0].real }, label: "Re P[0]")
        plt.plot(x: tSpace, y: trajectory.map { $0[0].imaginary }, label: "Im P[0]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[0].real }, label: "Re P1[0]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[0].imaginary }, label: "Im P1[0]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[0].real }, label: "Re P2[0]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[0].imaginary }, label: "Im P2[0]")
        plt.legend()
        plt.show()
        plt.close()
        
        plt.figure()
        plt.plot(x: tSpace, y: trajectory.map { $0[1].real }, label: "Re P[1]")
        plt.plot(x: tSpace, y: trajectory.map { $0[1].imaginary }, label: "Im P[1]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[1].real }, label: "Re P1[1]")
        plt.plot(x: tauSpace1, y: ket1.map { $0[1].imaginary }, label: "Im P1[1]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[1].real }, label: "Re P2[1]")
        plt.plot(x: tauSpace2, y: ket2.map { $0[1].imaginary }, label: "Im P2[1]")
        plt.legend()
        plt.show()
        plt.close()
        
        let expO1 = zip(normalizationFactor1, zip(bra1, ket1)).map { normSquared, braKet in
            let bra = braKet.0
            let ket = braKet.1
            return bra.inner(ket, metric: O) / normSquared
        }
        let expO2 = zip(normalizationFactor2, zip(bra2, ket2)).map { normSquared, braKet in
            let bra = braKet.0
            let ket = braKet.1
            return bra.inner(ket, metric: O) / normSquared
        }
        let expO = zip(tauSpace1, zip(expO1, expO2)).map { (tau, O1O2) in
            if tau < s {
                return 0.5 * (O1O2.0 + O1O2.1)
            } else {
                return O1O2.0 + O1O2.1
            }
        }
        plt.figure()
        plt.plot(x: tSpace, y: expectationValue.real, label: "Re <O>")
        plt.plot(x: tSpace, y: expectationValue.imaginary, label: "Im <O>")
        plt.plot(x: tauSpace1, y: expO.real, label: "Re (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.plot(x: tauSpace1, y: expO.imaginary, label: "Im (<O(t)O1(s)> + <O(t)O2(s)>)", linestyle: "--")
        plt.title("Non-linear s=\(s)")
        plt.legend()
        plt.show()
        plt.close()
    }
}
