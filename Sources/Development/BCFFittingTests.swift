//
//  BCFFittingTests.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 7.6.2026.
//

import HOPS
import PythonKit
import PythonKitUtilities
import SebbuScience
import NumericsExtensions
import Numerics

fileprivate func reconstructFunction(t: [Double], G: [Complex<Double>], W: [Complex<Double>]) -> [Complex<Double>] {
    t.map { t in
        var result: Complex<Double> = .zero
        for (g, w) in zip(G, W) {
            result += g * .exp(-w * t)
        }
        return result
    }
}

fileprivate func ____bathCorrelationFunction(t: Double, spectralDensity: (Double) -> Double) -> Complex<Double> {
    Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensity(omega) * .init(length: 1, phase: -omega * t)
    }
}

fileprivate func testArbitraryBCF() {
    let t: [Double] = .linearSpace(0, 20, 500)
    let bcf = t.map { t in
        Complex(Double.tanh(t) * .exp(-0.1 * t) + .sin(t * 3) * .cos(t * 2) * .exp(-0.21 * t))
    }
    let terms = 6
    let (G1, W1) = MatrixPencil.fit(y: bcf, dt: t[1] - t[0], terms: terms)
    let (G2, W2) = NonLinearFit.fit(t: t, y: bcf, terms: terms)
    
    let matrixPencilBcf = reconstructFunction(t: t, G: G1, W: W1)
    let nonLinearBcf = reconstructFunction(t: t, G: G2, W: W2)
    plt.figure()
    plt.plot(x: t, y: bcf.real, label: "Re Exact")
    plt.plot(x: t, y: matrixPencilBcf.real, label: "Re Matrix pencil")
    plt.plot(x: t, y: nonLinearBcf.real, label: "Re Non-linear", linestyle: "--")

    plt.plot(x: t, y: bcf.imaginary, label: "Im Exact")
    plt.plot(x: t, y: matrixPencilBcf.imaginary, label: "Im Matrix pencil")
    plt.plot(x: t, y: nonLinearBcf.imaginary, label: "Im Non-linear", linestyle: "--")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("bcf")
    plt.show()
    plt.close()
}

fileprivate func testPhysicalBCF() {
    let t: [Double] = .linearSpace(0, 10, 500)
    let w: [Double] = .linearSpace(0, 5, 500)
    let J = w.map { omega in
        let omega = omega.magnitude
        return 0.027 * omega * omega * omega * .exp(-omega * omega / (1.447 * 1.447))
    }
    
    let bcf = t.map { ____bathCorrelationFunction(t: $0) { omega in
        let omega = omega.magnitude
        return 0.027 * omega * omega * omega * .exp(-omega * omega / (1.447 * 1.447))
    }}
    let terms = 3
    let (G1, W1) = MatrixPencil.fit2(y: bcf, dt: t[1] - t[0], terms: terms)
    let (G2, W2, r) = NonLinearFit.fitPhysical(t: t, y: bcf, terms: terms)
    print()
    print(G1)
    print(G2)
    print(r)
    print()
    print(W1)
    print(W2)

    let matrixPencilBcf = reconstructFunction(t: t, G: G1, W: W1)
    let nonLinearBcf = reconstructFunction(t: t, G: G2, W: W2)
    plt.figure()
    plt.plot(x: t, y: bcf.real, label: "Re Exact")
    plt.plot(x: t, y: matrixPencilBcf.real, label: "Re Matrix pencil")
    plt.plot(x: t, y: nonLinearBcf.real, label: "Re Non-linear", linestyle: "--")

    plt.plot(x: t, y: bcf.imaginary, label: "Im Exact")
    plt.plot(x: t, y: matrixPencilBcf.imaginary, label: "Im Matrix pencil")
    plt.plot(x: t, y: nonLinearBcf.imaginary, label: "Im Non-linear", linestyle: "--")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("bcf")
    plt.show()
    plt.close()
    
    let matrixPencilBCFSpline = CubicHermiteSpline(x: t, y: matrixPencilBcf)
    let nonLinearBCFSpline = CubicHermiteSpline(x: t, y: nonLinearBcf)
    
    let JMatrixPencil = w.map { omega in
        Quad.integrate(a: 0, b: t.last!) { t in
            matrixPencilBCFSpline.sample(t) * .init(length: 1.0 / (.pi), phase: omega * t)
        }
    }
    
    let JNonLinear = w.map { omega in
        Quad.integrate(a: 0, b: t.last!) { t in
            nonLinearBCFSpline.sample(t) * .init(length: 1.0 / (.pi), phase: omega * t)
        }
    }
    
    plt.figure()
    plt.plot(x: w, y: J, label: "Exact")
    plt.plot(x: w, y: JMatrixPencil.real, label: "MP")
    plt.plot(x: w, y: JNonLinear.real, label: "JNonLinear")
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("J")
    plt.show()
    plt.close()
    
}

func testBCFFittings() {
//    testArbitraryBCF()
    testPhysicalBCF()
}

