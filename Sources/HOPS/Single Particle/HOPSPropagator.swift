//
//  HOPSPropagator.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 20.10.2025.
//

import SebbuScience

public struct HOPSPropagator {
    @usableFromInline
    internal var solver: Solver
    
    @inlinable
    @_transparent
    public var t: Double {
        solver.t
    }
    
    @inlinable
    @_transparent
    public var currentState: [Vector<Complex<Double>>] {
        solver.currentState
    }
    
    @inlinable
    internal init(_ solver: RK45FixedStep<[Vector<Complex<Double>>]>) {
        self.solver = .deterministic(.init(solver))
    }
    
    @inlinable
    internal init(_ solver: SRK2FixedStep<[Vector<Complex<Double>>], Complex<Double>>) {
        self.solver = .stochasticSingleNoise(.init(solver))
    }
    
    @inlinable
    internal init(_ solver: SRK2FixedStepMultiNoise<[Vector<Complex<Double>>], Complex<Double>>) {
        self.solver = .stochasticMultiNoise(.init(solver))
    }
    
    @inlinable
    @inline(__always)
    public mutating func step() -> (t: Double, state: [Vector<Complex<Double>>]) {
        solver.step()
    }
    
    @inlinable
    @inline(__always)
    public mutating func reset(initialState: [Vector<Complex<Double>>], t0: Double) {
        solver.reset(initialState: initialState, t0: t0)
    }
    
}

extension HOPSPropagator {
    @usableFromInline
    internal final class SolverBox<T> {
        @usableFromInline
        var solver: T
        
        @inlinable
        @inline(__always)
        internal init(_ solver: T) {
            self.solver = solver
        }
    }
    
    @usableFromInline
    enum Solver {
        case deterministic(SolverBox<RK45FixedStep<[Vector<Complex<Double>>]>>)
        case stochasticSingleNoise(SolverBox<SRK2FixedStep<[Vector<Complex<Double>>], Complex<Double>>>)
        case stochasticMultiNoise(SolverBox<SRK2FixedStepMultiNoise<[Vector<Complex<Double>>], Complex<Double>>>)
        
        @inlinable
        @inline(__always)
        var t: Double {
            switch self {
            case .deterministic(let box): return box.solver.t
            case .stochasticSingleNoise(let box): return box.solver.t
            case .stochasticMultiNoise(let box): return box.solver.t
            }
        }
        
        @inlinable
        @inline(__always)
        var currentState: [Vector<Complex<Double>>] {
            switch self {
            case .deterministic(let box): return box.solver.currentState
            case .stochasticSingleNoise(let box): return box.solver.currentState
            case .stochasticMultiNoise(let box): return box.solver.currentState
            }
        }
        
        @inlinable
        @inline(__always)
        mutating func step() -> (Double, [Vector<Complex<Double>>]) {
            switch self {
            case .deterministic(let box):
                return box.solver.step()
            case .stochasticSingleNoise(let box):
                return box.solver.step()
            case .stochasticMultiNoise(let box):
                return box.solver.step()
            }
        }
        
        @inlinable
        @inline(__always)
        mutating func reset(initialState: [Vector<Complex<Double>>], t0: Double) {
            switch self {
            case .deterministic(let box):
                box.solver.reset(initialState: initialState, t0: t0)
            case .stochasticSingleNoise(let box):
                box.solver.reset(initialState: initialState, t0: t0)
            case .stochasticMultiNoise(let box):
                box.solver.reset(initialState: initialState, t0: t0)
            }
        }
    }
}
