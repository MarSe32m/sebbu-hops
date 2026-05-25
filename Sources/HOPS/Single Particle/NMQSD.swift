//
//  NMQSD.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 11.9.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public struct NMQSDCalculation: Sendable {
    public let dimension: Int
    public let L: Matrix<Complex<Double>>

    @usableFromInline
    internal let bcf: @Sendable (_ t: Double, _ s: Double) -> Complex<Double>

    @usableFromInline
    internal let G: [Complex<Double>]

    @usableFromInline
    internal let W: [Complex<Double>]

    /// Construct an NMQSDCalculation struct for trajectory computations
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: Coupling operator of the system
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>]) {
        self.dimension = dimension
        self.L = L
        self.bcf = { t, s in 
            var result: Complex<Double> = .zero
            for i in 0..<G.count {
                result = Relaxed.multiplyAdd(G[i], Complex.exp(-(t - s) * W[i]), result)
            }
            return result
        }
        self.G = G
        self.W = W
    }

    @inlinable
    public func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, OBar: (_ t: Double, _ z: Noise) -> Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(dimension), count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var _LDaggerOBar: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        return withoutActuallyEscaping(OBar) { OBar in 
            var solver = RK4Solver(initialState: initialState, t0: start, dt: stepSize) { t, psi in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                Heff.copyElements(from: iH)
                Heff.add(L, multiplied: z(t).conjugate)
                LDagger.dot(OBar(t, z), into: &_LDaggerOBar)
                Heff.subtract(_LDaggerOBar)
                Heff.dot(psi, into: &result)
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(Int((end - start) / stepSize) + 2)
            for _ in 0..<trajectory.capacity {
                trajectory.append(.zero(dimension))
            }
            var stateIndex = 0
            while solver.t < end {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory[stateIndex].copyComponents(from: state)
                stateIndex += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
    
    @inlinable
    public func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        //var stateVectorCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(dimension), count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        var DOperators: [Matrix<Complex<Double>>] = .init(repeating: .zeros(rows: H.rows, columns: H.columns), count: G.count)
        var DSum: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var trajectory: [Vector<Complex<Double>>] = []
        var tSpace: [Double] = []
        var t = start
        var state: Vector<Complex<Double>> = initialState
        //TODO: Optimize DOperator updating, and use RK45!
        //var DUpdate: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        while t < end {
            trajectory.append(state)
            tSpace.append(t)
            Heff.copyElements(from: iH)
            Heff.add(L, multiplied: z(t).conjugate)
            LDagger.dot(DOperators.first!, into: &DSum)
            for D in DOperators.dropFirst() {
                LDagger.dot(D, addingInto: &DSum)
            }
            Heff.subtract(DSum)
            state += Heff.dot(state, multiplied: Complex(stepSize))
            for i in DOperators.indices {
                DOperators[i] += stepSize * (G[i] * L - W[i] * DOperators[i] + Heff.dot(DOperators[i]) - DOperators[i].dot(Heff))
            }
            t += stepSize
        }
        return (tSpace, trajectory)
    }

    @inlinable
    public func solveLinear2<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], includePropagator: Bool = false, stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>], propagator: [Matrix<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        let _initialState: [Matrix<Complex<Double>>] = [.identity(rows: H.rows)] + .init(repeating: .zeros(rows: H.rows, columns: H.columns), count: G.count)
        var resultCache: Deque<[Matrix<Complex<Double>>]> = .init(repeating: _initialState, count: 4)
        var DSum: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var scratch: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var stateVector: Vector<Complex<Double>> = initialState
        var solver = RK4Solver(initialState: _initialState, t0: start, dt: stepSize) { t, currentState in
            var result = resultCache.removeFirst()
            defer { resultCache.append(result) }
            currentState[0].dot(initialState, into: &stateVector)
            Heff.copyElements(from: iH)
            Heff._add(L, multiplied: z(t).conjugate)
            for customOperator in customOperators {
                Heff._add(customOperator(t, stateVector))
            }
            DSum.zeroElements()
            for i in 1..<currentState.count {
                DSum.add(currentState[i])
                //LDagger._dot(currentState[i], addingInto: &DSum)
            }
            LDagger._dot(DSum, into: &scratch)
            Heff._subtract(scratch)
            Heff._dot(currentState[0], into: &result[0])
            for i in 1..<result.count {
                result[i].zeroElements()
                result[i].copyElements(from: L)
                result[i]._multiply(by: G[i - 1])
                result[i]._add(currentState[i], multiplied: -W[i - 1])
                Heff._dot(currentState[i], addingInto: &result[i])
                currentState[i]._dot(Heff, multiplied: -.one, addingInto: &result[i])
                //LDagger.dot(currentState[i], into: &scratch)
                //scratch.dot(DSum, addingInto: &result[i])
            }
//            for i in 1..<result.count {
//                for j in 1..<result.count {
//                    let C = commutator(currentState[i], currentState[j])
//                    if C.frobeniusNorm > 1e-12 {
//                        print(i, j, C.frobeniusNorm)
//                    }
//                }
//            }
            return result
        }
        
        var tSpace: [Double] = []
        var propagator: [Matrix<Complex<Double>>] = []
        var trajectory: [Vector<Complex<Double>>] = []
        if includePropagator {
            for _ in 0..<Int((end - start) / stepSize) + 1 {
                propagator.append(.zeros(rows: H.rows, columns: H.columns))
            }
        }
        for _ in 0..<Int((end - start) / stepSize) + 1 {
            trajectory.append(.zero(initialState.count))
        }
        var index = 0
        while solver.t < end {
            let (t, state) = solver.step()
            tSpace.append(t)
            if index >= trajectory.count {
                trajectory.append(.zero(initialState.count))
            }
            state[0].dot(initialState, into: &trajectory[index])
            
            if includePropagator {
                if index >= propagator.count {
                    propagator.append(.identity(rows: H.rows))
                }
                propagator[index].copyElements(from: state[0])
            }
            index += 1
        }
        return (tSpace, trajectory, propagator)
    }
    
    @inlinable
    public func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, OBar: (_ t: Double, _ z: Noise) -> Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(dimension), .zero(self.G.count)], count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var _LDaggerOBar: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        let shiftVectorInitial: Vector<Complex<Double>> = .zero(self.G.count)
        return withoutActuallyEscaping(OBar) { OBar in 
            var solver = RK4Solver(initialState: [initialState, shiftVectorInitial], t0: start, dt: stepSize) { t, psi in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                let currentState = psi[0]
                let currentShiftVector = psi[1]
                let LDaggerExpectation = currentState.inner(currentState, metric: LDagger) / (currentState.normSquared + 1e-12)
                var noiseShift: Complex<Double> = .zero
                for i in 0..<currentShiftVector.count {
                    noiseShift = Relaxed.sum(noiseShift, currentShiftVector[i])
                    result[1][i] = G[i].conjugate * LDaggerExpectation - W[i].conjugate * currentShiftVector[i]
                }

                Heff.copyElements(from: iH)
                Heff._add(L, multiplied: z(t).conjugate + noiseShift)
                let _OBar = OBar(t, z)
                LDagger._dot(_OBar, into: &_LDaggerOBar)
                Heff._add(_OBar, multiplied: LDaggerExpectation)
                Heff._subtract(_LDaggerOBar)
                Heff._dot(psi[0], into: &result[0])
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 1)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(Int((end - start) / stepSize) + 1)
            for _ in 0..<trajectory.capacity {
                trajectory.append(.zero(dimension))
            }
            var stateIndex = 0
            while solver.t < end {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory[stateIndex].copyComponents(from: state[0])
                stateIndex += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
    
    @inlinable
    public func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        //var stateVectorCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(dimension), count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let identity: Matrix<Complex<Double>> = .identity(rows: H.rows)
        var shiftVector: Vector<Complex<Double>> = .zero(G.count)
        var shiftedLDagger: Matrix<Complex<Double>> = identity
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        var DOperators: [Matrix<Complex<Double>>] = .init(repeating: .zeros(rows: H.rows, columns: H.columns), count: G.count)
        var DSum: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var trajectory: [Vector<Complex<Double>>] = []
        var tSpace: [Double] = []
        var t = start
        var state: Vector<Complex<Double>> = initialState
        //TODO: Optimize DOperator updating, and use RK45!
        //var DUpdate: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        while t < end {
            trajectory.append(state)
            tSpace.append(t)
            var shift: Complex<Double> = .zero
            for i in 0..<shiftVector.count {
                shift += shiftVector[i]
            }
            let LDaggerExp = state.inner(state, metric: LDagger) / (state.normSquared + 1e-12)
            shiftedLDagger.zeroElements()
            shiftedLDagger.add(LDagger)
            shiftedLDagger.subtract(identity, multiplied: LDaggerExp)
            for i in 0..<shiftVector.count {
                shiftVector[i] += stepSize * (G[i].conjugate * LDaggerExp - W[i].conjugate * shiftVector[i])
            }
            Heff.copyElements(from: iH)
            Heff.add(L, multiplied: z(t).conjugate + shift)
            shiftedLDagger._dot(DOperators.first!, into: &DSum)
            for D in DOperators.dropFirst() {
                shiftedLDagger._dot(D, addingInto: &DSum)
            }
            Heff.subtract(DSum)
            state += Heff.dot(state, multiplied: Complex(stepSize))
            for i in DOperators.indices {
                DOperators[i] += stepSize * (G[i] * L - W[i] * DOperators[i] + Heff.dot(DOperators[i]) - DOperators[i].dot(Heff))
            }
            t += stepSize
        }
        return (tSpace, trajectory)
    }
    
    @inlinable
    func commutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>) -> Matrix<Complex<Double>> {
        A.dot(B) - B.dot(A)
    }
    
    @inlinable
    public func solveNonLinear2<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], includePropagator: Bool = false, stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>], propagator: [Matrix<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let identity: Matrix<Complex<Double>> = .identity(rows: H.rows)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        var shiftedLDagger: Matrix<Complex<Double>> = identity
        var stateVector: Vector<Complex<Double>> = .zero(initialState.count)
        // [U, shift] + [D_1, D_2, ...]
        let _initialState: [Matrix<Complex<Double>>] = [.identity(rows: H.rows), .zeros(rows: 1, columns: G.count)] + .init(repeating: .zeros(rows: H.rows, columns: H.columns), count: G.count)
        var resultCache: Deque<[Matrix<Complex<Double>>]> = .init(repeating: _initialState, count: 4)
        var DSum: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var scratch: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var solver = RK4Solver(initialState: _initialState, t0: start, dt: stepSize) { t, currentState in
            var result = resultCache.removeFirst()
            defer { resultCache.append(result) }
            var noiseShift: Complex<Double> = .zero
            for i in 0..<currentState[1].elements.count {
                noiseShift += currentState[1].elements[i]
            }
            currentState[0]._dot(initialState, into: &stateVector)
            let LDaggerExp = stateVector.inner(stateVector, metric: LDagger) / (stateVector.normSquared + 1e-12)
            for i in 0..<currentState[1].elements.count {
                result[1].elements[i] = G[i].conjugate * LDaggerExp - W[i].conjugate * currentState[1].elements[i]
            }
            shiftedLDagger.copyElements(from: LDagger)
            shiftedLDagger._add(identity, multiplied: -LDaggerExp)
            Heff.copyElements(from: iH)
            Heff._add(L, multiplied: z(t).conjugate + noiseShift)
            for customOperator in customOperators {
                Heff._add(customOperator(t, stateVector))
            }
            DSum.zeroElements()
            for i in 2..<currentState.count {
                DSum.add(currentState[i])
                //shiftedLDagger._dot(currentState[i], addingInto: &DSum)
            }
            shiftedLDagger._dot(DSum, into: &scratch)
            Heff._subtract(scratch)
            Heff._dot(currentState[0], into: &result[0])
            for i in 2..<result.count {
                result[i].copyElements(from: L)
                result[i]._multiply(by: G[i - 2])
                result[i]._add(currentState[i], multiplied: -W[i - 2])
                Heff._dot(currentState[i], addingInto: &result[i])
                currentState[i]._dot(Heff, multiplied: -.one, addingInto: &result[i])
//                shiftedLDagger.dot(currentState[i], into: &scratch)
//                scratch.dot(DSum, multiplied: -.one, addingInto: &result[i])
            }
//            for i in 2..<result.count {
//                for j in 2..<result.count {
//                    let C = commutator(currentState[i], currentState[j])
//                    if C.frobeniusNorm > 1e-12 {
//                        print(i, j, C.frobeniusNorm, currentState[i].frobeniusNorm, currentState[j].frobeniusNorm)
//                    }
//                }
//            }
            return result
        }
        
        var tSpace: [Double] = []
        var propagator: [Matrix<Complex<Double>>] = []
        var trajectory: [Vector<Complex<Double>>] = []
        if includePropagator {
            for _ in 0..<Int((end - start) / stepSize) + 1 {
                propagator.append(.zeros(rows: H.rows, columns: H.columns))
            }
        }
        for _ in 0..<Int((end - start) / stepSize) + 1 {
            trajectory.append(.zero(initialState.count))
        }
        var index = 0
        while solver.t < end {
            let (t, state) = solver.step()
            tSpace.append(t)
            if index >= trajectory.count {
                trajectory.append(.zero(initialState.count))
            }
            state[0].dot(initialState, into: &trajectory[index])
            
            if includePropagator {
                if index >= propagator.count {
                    propagator.append(.identity(rows: H.rows))
                }
                propagator[index].copyElements(from: state[0])
            }
            index += 1
        }
        return (tSpace, trajectory, propagator)
        
//        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
//        let identity: Matrix<Complex<Double>> = .identity(rows: H.rows)
//        var shiftVector: Vector<Complex<Double>> = .zero(G.count)
//        var shiftedLDagger: Matrix<Complex<Double>> = identity
//        let LDagger = L.conjugateTranspose
//        let iH = -.i * H
//        var DOperators: [Matrix<Complex<Double>>] = .init(repeating: .zeros(rows: H.rows, columns: H.columns), count: G.count)
//        var DSum: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
//        var trajectory: [Vector<Complex<Double>>] = []
//        var tSpace: [Double] = []
//        var t = start
//        var state: Vector<Complex<Double>> = initialState
//        //TODO: Optimize DOperator updating, and use RK45!
//        //var DUpdate: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
//        while t < end {
//            trajectory.append(state)
//            tSpace.append(t)
//            var shift: Complex<Double> = .zero
//            for i in 0..<shiftVector.count {
//                shift += shiftVector[i]
//            }
//            let LDaggerExp = state.inner(state, metric: LDagger) / (state.normSquared + 1e-12)
//            shiftedLDagger.zeroElements()
//            shiftedLDagger.add(LDagger)
//            shiftedLDagger.subtract(identity, multiplied: LDaggerExp)
//            for i in 0..<shiftVector.count {
//                shiftVector[i] += stepSize * (G[i].conjugate * LDaggerExp - W[i].conjugate * shiftVector[i])
//            }
//            Heff.copyElements(from: iH)
//            Heff.add(L, multiplied: z(t).conjugate + shift)
//            shiftedLDagger._dot(DOperators.first!, into: &DSum)
//            for D in DOperators.dropFirst() {
//                shiftedLDagger._dot(D, addingInto: &DSum)
//            }
//            Heff.subtract(DSum)
//            state += Heff.dot(state, multiplied: Complex(stepSize))
//            for i in DOperators.indices {
//                DOperators[i] += stepSize * (G[i] * L - W[i] * DOperators[i] + Heff.dot(DOperators[i]) - DOperators[i].dot(Heff))
//            }
//            t += stepSize
//        }
//        return (tSpace, trajectory)
    }
    
    /// Maps a NMQSD trajectory to the corresponding density matrix
    /// - Parameters:
    ///   - trajectory: The NMSD trajectory to map to density matrix
    ///   - normalized: Whether the trajectory should be normalized. Default value is false.
    /// - Returns: Array of density matrices
    @inlinable
    public func mapTrajectoryToDensityMatrix(_ trajectory: [Vector<Complex<Double>>], normalize: Bool = false) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
            if normalize { rho[rho.count - 1] /= state.normSquared }
        }
        return rho
    }
}
