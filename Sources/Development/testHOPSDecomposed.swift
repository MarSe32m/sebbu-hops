//
//  TwoTimeCorrelation.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 7.10.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

public func testHOPSDecomposedPropagation(endTime: Double = 7.0, at: Double) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 500)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .one], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 3)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let z = GaussianFFTNoiseProcess(tMax: endTime, dtMax: 0.01) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    // Original HOPS trajectory
    let (hopsOriginalTSpace, hopsOriginalTrajectory) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
    
    let O1: Matrix<Complex<Double>> = .init(elements: [
        .zero, .one,
        .zero, .zero
    ], rows: 2, columns: 2)
    // Decomposed HOPS trajectory
    let (hopsDecomposedTSpace1, hopsDecomposedTrajectory1) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, O: O1, at: at)
    
    let O2: Matrix<Complex<Double>> = .identity(rows: 2) - O1
    // Decomposed HOPS trajectory
    let (hopsDecomposedTSpace2, hopsDecomposedTrajectory2) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, O: O2, at: at)
    
    let spline1 = CubicHermiteSpline(x: hopsDecomposedTSpace1, y: hopsDecomposedTrajectory1)
    let spline2 = CubicHermiteSpline(x: hopsDecomposedTSpace2, y: hopsDecomposedTrajectory2)
    
    let hopsDecomposedTSpace = hopsDecomposedTSpace1
    let hopsDecomposedTrajectory: [Vector<Complex<Double>>] = hopsDecomposedTSpace.map { $0 < at ? 0.5 * (spline1.sample($0) + spline2.sample($0)) : (spline1.sample($0) + spline2.sample($0)) }

    plt.figure()
    //plt.plot(x: nmqsdTSpace, y: nmqsdTrajectory.map { $0[0].real }, label: "NMQSD Re C_g(t)")
    plt.plot(x: hopsOriginalTSpace, y: hopsOriginalTrajectory.map { $0[0].real }, label: "HOPS Re C_g(t)")
    plt.plot(x: hopsDecomposedTSpace, y: hopsDecomposedTrajectory.map { $0[0].real }, label: "HOPS dec Re C_g(t)", linestyle: "-.")
    
    //plt.plot(x: nmqsdTSpace, y: nmqsdTrajectory.map { $0[0].imaginary }, label: "NMQSD Im C_g(t)")
    plt.plot(x: hopsOriginalTSpace, y: hopsOriginalTrajectory.map { $0[0].imaginary }, label: "HOPS Im C_g(t)")
    plt.plot(x: hopsDecomposedTSpace, y: hopsDecomposedTrajectory.map { $0[0].imaginary }, label: "HOPS dec Im C_g(t)", linestyle: "-.")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("C")
    plt.show()
    plt.close()
    
    plt.figure()
    //plt.plot(x: nmqsdTSpace, y: nmqsdTrajectory.map { $0[1].real }, label: "NMQSD Re C_x(t)")
    plt.plot(x: hopsOriginalTSpace, y: hopsOriginalTrajectory.map { $0[1].real }, label: "HOPS Re C_x(t)")
    plt.plot(x: hopsDecomposedTSpace, y: hopsDecomposedTrajectory.map { $0[1].real }, label: "HOPS dec Re C_x(t)", linestyle: "-.")
    
    //plt.plot(x: nmqsdTSpace, y: nmqsdTrajectory.map { $0[1].imaginary }, label: "NMQSD Im C_x(t)")
    plt.plot(x: hopsOriginalTSpace, y: hopsOriginalTrajectory.map { $0[1].imaginary }, label: "HOPS Im C_x(t)")
    plt.plot(x: hopsDecomposedTSpace, y: hopsDecomposedTrajectory.map { $0[1].imaginary }, label: "HOPS dec Im C_x(t)", linestyle: "-.")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("C")
    plt.show()
    plt.close()
}
