//
//  UnifiedHOPSStateFunc.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 12.6.2026.
//

extension UnifiedHOPSHierarchy {
    @usableFromInline
    internal protocol HOPSStateFuncProtocol: ~Copyable, ~Escapable {
        associatedtype State: ~Copyable & HOPSStateProtocol
        
        @inlinable
        func zero() -> State
    }
}
