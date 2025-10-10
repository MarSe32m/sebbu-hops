//
//  QFunctionPlot.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 25.9.2025.
//

import HOPS
import SebbuScience
import SebbuCollections
import PythonKit
import PythonKitUtilities

private func lorentzian(A: Double, omegaC: Double, gamma: Double, omega: Double) -> Double {
    return A * gamma / (Double.pow(omega - omegaC, 2) + gamma * gamma)
}

private func _bcf(t: Double, spectralDensity: (Double) -> Double) -> Complex<Double> {
    return Quad.integrate(a: -.infinity, b: .infinity) { omega in
        Complex(length: spectralDensity(omega), phase: -omega * t)
    }
}

public func QFunctionPlot(endTime: Double) {
    let A = 0.1
    let omegaC = 1.0
    let gamma = 0.01
    
    let (G, W) = {
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { _bcf(t: $0) { omega in
            lorentzian(A: A, omegaC: omegaC, gamma: gamma, omega: omega)
        } }
        //let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 1)
        let G = [Complex(.pi * A)]
        let W = [Complex(gamma, omegaC)]
//        plt.figure()
//        plt.plot(x: tSpace, y: bcf.real)
//        plt.plot(x: tSpace, y: bcf.imaginary)
//        plt.plot(x: tSpace, y: tSpace.map {G[0] * .exp(-$0 * W[0]) }.real, linestyle: "--")
//        plt.plot(x: tSpace, y: tSpace.map {G[0] * .exp(-$0 * W[0]) }.imaginary, linestyle: "--")
//        plt.show()
//        plt.close()
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy1 = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    let hierarchy2 = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 3)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime, dtMax: 0.001) { omega in
       lorentzian(A: A, omegaC: omegaC, gamma: gamma, omega: omega)
    }
    
    let noise = zGenerator.generate()
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    let (linearTSpace, linearTrajectory) = hierarchy1.solveLinear(end: endTime, initialState: initialState, H: H, z: noise, stepSize: 0.01, includeHierarchy: true)
    let (nonLinearTSpace, nonLinearTrajectory, _) = hierarchy1.solveNonLinear(end: endTime, initialState: initialState, H: H, z: noise, shiftType: .none, stepSize: 0.01, includeHierarchy: true)
    let (nonLinearMeanShiftedTSpace, nonLinearMeanShiftedTrajectory, shift) = hierarchy2.solveNonLinear(end: endTime, initialState: initialState, H: H, z: noise, shiftType: .meanField, stepSize: 0.01, includeHierarchy: true)
    let (nonLinearExactShiftedTSpace, nonLinearExactShiftedTrajectory, exactShift) = hierarchy2.solveNonLinearShifted(end: endTime, initialState: initialState, H: H, z: noise, stepSize: 0.01, includeHierarchy: true)
    
    let nonLinearMeanShifted = hierarchy1.mapLinearToDensityMatrix( nonLinearMeanShiftedTrajectory.map {Vector(Array($0.components[0..<2]))})
    let nonLinearExactShiftedRho = hierarchy1.mapLinearToDensityMatrix(nonLinearExactShiftedTrajectory.map {Vector(Array($0.components[0..<2]))})
    let nonLinearRho = hierarchy1.mapLinearToDensityMatrix(nonLinearTrajectory.map {Vector(Array($0.components[0..<2]))})

    plt.figure()
    plt.plot(x: nonLinearTSpace, y: nonLinearRho.map { $0[0, 0] - $0[1, 1] }.real, label: "NLS: <z>")
    plt.plot(x: nonLinearTSpace, y: nonLinearRho.map { 2 * $0[0, 1].real }, label: "NLS: <x>")
    plt.plot(x: nonLinearTSpace, y: nonLinearRho.map { 2 * $0[0, 1].imaginary } , label: "NLS: <y>")
    
    plt.plot(x: nonLinearMeanShiftedTSpace, y: nonLinearMeanShifted.map { $0[0, 0] - $0[1, 1] }.real, label: "NLM: <z>", linestyle: "-.")
    plt.plot(x: nonLinearMeanShiftedTSpace, y: nonLinearMeanShifted.map { 2 * $0[0, 1].real }, label: "NLM: <x>", linestyle: "-.")
    plt.plot(x: nonLinearMeanShiftedTSpace, y: nonLinearMeanShifted.map { 2 * $0[0, 1].imaginary } , label: "NLM: <y>", linestyle: "-.")
    
    plt.plot(x: nonLinearExactShiftedTSpace, y: nonLinearExactShiftedRho.map { $0[0, 0] - $0[1, 1] }.real, label: "NLE: <z>", linestyle: "--")
    plt.plot(x: nonLinearExactShiftedTSpace, y: nonLinearExactShiftedRho.map { 2 * $0[0, 1].real }, label: "NLE: <x>", linestyle: "--")
    plt.plot(x: nonLinearExactShiftedTSpace, y: nonLinearExactShiftedRho.map { 2 * $0[0, 1].imaginary } , label: "NLE: <y>", linestyle: "--")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<O>")
    plt.show()
    plt.close()
    
    
    //plt.figure()
    //plt.plot(x: nonLinearMeanShiftedTSpace, y: shift.real, label: "Re u(t)")
    //plt.plot(x: nonLinearMeanShiftedTSpace, y: shift.imaginary, label: "Im u(t)")
    
    //plt.plot(x: nonLinearExactShiftedTSpace, y: exactShift.real, label: "Re exact u(t)")
    //plt.plot(x: nonLinearExactShiftedTSpace, y: exactShift.imaginary, label: "Im exact u(t)")
    
    //plt.plot(x: nonLinearTSpace, y: LDaggerExp.real, label: "Re <Ld>", linestyle: "--")
    //plt.plot(x: nonLinearTSpace, y: LDaggerExp.imaginary, label: "Im <Ld>", linestyle: "--")
    //plt.legend()
    //plt.xlabel("t")
    //plt.ylabel("u(t)")
    //plt.show()
    //plt.close()
    
    //plt.figure()
    //plt.plot(x: nonLinearMeanShiftedTSpace, y: nonLinearMeanShiftedTSpace.enumerated().map { i, t in noise(t).conjugate + shift[i] }.real, label: "Re shifted z(t)^*" )
    //plt.plot(x: nonLinearMeanShiftedTSpace, y: nonLinearMeanShiftedTSpace.enumerated().map { i, t in noise(t).conjugate + shift[i] }.imaginary, label: "Im shifted z(t)^*" )
    //plt.legend()
    //plt.xlabel("t")
    //plt.ylabel("z^*(t)")
    //plt.show()
    //plt.close()
    
    let linearQFunction = QFunction(tSpace: linearTSpace, totalStates: linearTrajectory, dimension: 2)
    let nonLinearQFunction = QFunction(tSpace: nonLinearTSpace, totalStates: nonLinearTrajectory, dimension: 2)
    let nonLinearMeanShiftQFunction = QFunction(tSpace: nonLinearMeanShiftedTSpace, totalStates: nonLinearMeanShiftedTrajectory, dimension: 2)
    let nonLinearExactShiftQFunction = QFunction(tSpace: nonLinearExactShiftedTSpace, totalStates: nonLinearExactShiftedTrajectory, dimension: 2)
    
    let xAxis = [Double].linearSpace(-10, 10.5, 200)
    let yAxis = [Double].linearSpace(-10, 10.5, 200)
    let tSpace = [Double].linearSpace(0, endTime, Int(endTime / 0.1))
    
    plt.figure()
    plt.plot(x: tSpace, y: tSpace.map { linearQFunction.numberOperatorExpectationValue(t: $0) }, label: "Linear")
    plt.plot(x: tSpace, y: tSpace.map { nonLinearQFunction.numberOperatorExpectationValue(t: $0) }, label: "Non-linear")
    plt.plot(x: tSpace, y: tSpace.map { nonLinearMeanShiftQFunction.numberOperatorExpectationValue(t: $0) }, label: "Non-linear mean", linestyle: "--")
    plt.plot(x: tSpace, y: tSpace.map { nonLinearExactShiftQFunction.numberOperatorExpectationValue(t: $0) }, label: "Non-linear exact", linestyle: "-.")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<n>")
    plt.show()
    plt.close()

    _plotQFunction(linearQFunction, xAxis: xAxis, yAxis: yAxis, tSpace: tSpace, title: "Linear")
    _plotQFunction(nonLinearQFunction, xAxis: xAxis, yAxis: yAxis, tSpace: tSpace, title: "Non-linear")
    _plotQFunction(nonLinearMeanShiftQFunction, xAxis: xAxis, yAxis: yAxis, tSpace: tSpace, title: "Non-linear mean")
    _plotQFunction(nonLinearExactShiftQFunction, xAxis: xAxis, yAxis: yAxis, tSpace: tSpace, title: "Non-linear exact", show: true)
    //_plotAuxiliaryStateTarjectories(dimension: 2, tSpace: nonLinearExactShiftedTSpace, trajectory: nonLinearExactShiftedTrajectory, title: "Non-linear exact shift", hierarchy: hierarchy)
    //_plotAuxiliaryStateTarjectories(dimension: 2, tSpace: nonLinearMeanShiftedTSpace, trajectory: nonLinearMeanShiftedTrajectory, title: "Non-linear mean shift", hierarchy: hierarchy)
    //_plotAuxiliaryStateTarjectories(dimension: 2, tSpace: linearTSpace, trajectory: linearTrajectory, title: "Linear", hierarchy: hierarchy)
}

private func _factorial(_ n: Int) -> Double {
    if n <= 1 { return 1.0 }
    var result = 1.0
    for i in 1...n {
        result *= Double(i)
    }
    return result
}

struct QFunction {
    let stateSpline: CubicHermiteSpline<Vector<Complex<Double>>>
    let dimension: Int
    
    init(tSpace: [Double], totalStates: [Vector<Complex<Double>>], dimension: Int) {
        self.stateSpline = CubicHermiteSpline(x: tSpace, y: totalStates)
        self.dimension = dimension
    }
    
    func callAsFunction(t: Double, alpha: Complex<Double>) -> Double {
        let totalState = stateSpline.sample(t)
        var alphaPsi: Vector<Complex<Double>> = .zero(dimension)
        for i in stride(from: 0, to: totalState.count, by: dimension) {
            let k = i / dimension
            let auxiliaryState = Vector(Array(totalState.components[i..<i+dimension]))
            let factorial = _factorial(k).squareRoot()
            alphaPsi.add(auxiliaryState, multiplied: .pow(alpha.conjugate, k) / factorial)
        }
        return .exp(-alpha.lengthSquared) * alphaPsi.normSquared / (.pi * totalState.normSquared)
    }
    
    func numberOperatorExpectationValue(t: Double) -> Double {
        let totalState = stateSpline.sample(t)
        let normSquared = totalState.normSquared
        var result: Double = .zero
        for i in stride(from: 0, to: totalState.count, by: dimension) {
            let k = i / dimension
            let auxiliaryState = Vector(Array(totalState.components[i..<i+dimension]))
            result += Double(k) * auxiliaryState.normSquared
        }
        return result / normSquared
    }
}

private func _plotQFunction(_ Q: QFunction, xAxis: [Double], yAxis: [Double], tSpace: [Double], title: String, show: Bool = false) {
    let data = tSpace.parallelMap { t in
        xAxis.betterMap { x in
            yAxis.betterMap { y in
                Q(t: t, alpha: Complex(x, y))
            }
        }
    }
    let (fig, ax) = plt.subplots()
    _plt.subplots_adjust(bottom: 0.25)
    let heatMap = ax.imshow(data[0], origin: "lower", cmap: "viridis", extent: [Int(xAxis.min()!), Int(xAxis.max()!), Int(yAxis.min()!), Int(yAxis.max()!)], aspect: "auto")
    let bar = _plt.colorbar(heatMap, ax: ax)
    ax.set_title("Q-function \(title), <n>=0")
    
    let axSlider = _plt.axes([0.2, 0.1, 0.6, 0.03])
    let slider = _wdg.Slider(axSlider, "t index", 0, tSpace.count - 1, valinit: 0, valstep: 1)
    
    let update = PythonFunction { val in
        let idx = Int(slider.val)!
        heatMap.set_data(data[idx])
        let newData = data[idx]
        var min = 1.0 / .pi
        var max = 0.0
        for xAxis in newData {
            min = Swift.min(min, xAxis.min()!)
            max = Swift.max(max, xAxis.max()!)
        }
        heatMap.set_clim(vmin: min, vmax: max)
        let nExpString = String(format: "%.3f", Q.numberOperatorExpectationValue(t: tSpace[idx]))
        ax.set_title("Q-function \(title), <n>=\(nExpString)")
        bar.update_normal(heatMap)
        fig.canvas.draw_idle()
        return 0
    }
    
    slider.on_changed(update)
    if show { plt.show() }
}

private func _plotAuxiliaryStateTarjectories(dimension: Int, tSpace: [Double], trajectory: [Vector<Complex<Double>>], title: String, hierarchy: HOPSHierarchy) {
    print("\(title) hierarchy size: \(trajectory[0].count / 2)")
    for i in stride(from: 0, to: trajectory[0].count, by: 2) {
        let states = trajectory.map { Vector(Array($0.components[i..<i+2])) }
        let rho = hierarchy.mapLinearToDensityMatrix(states)
        let Z = rho.map { $0[0, 0].real - $0[1, 1].real }
        let X = rho.map { 2 * $0[0, 1].real }
        let Y = rho.map { 2 * $0[0, 1].imaginary }
        plt.figure()
        plt.plot(x: tSpace, y: Z, label: "Z")
        plt.plot(x: tSpace, y: X, label: "X")
        plt.plot(x: tSpace, y: Y, label: "Y")
        plt.title("\(title): Auxiliary state \(i/2)")
        plt.legend()
        plt.show()
        plt.close()
    }
}
