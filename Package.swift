// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "sebbu-hops",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "HOPS", targets: ["HOPS"]),
    ],
    dependencies: [
        .package(url: "https://github.com/MarSe32m/sebbu-science", branch: "main"),
        .package(url: "https://github.com/apple/swift-algorithms", branch: "main")
    ],
    targets: [
        .target(
            name: "HOPS",
            dependencies: [.product(name: "SebbuScience", package: "sebbu-science"),
                           .product(name: "Algorithms", package: "swift-algorithms")],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .executableTarget(
            name: "Development",
            dependencies: [
                .target(name: "HOPS"),
                .product(name: "SebbuScience", package: "sebbu-science"),
                .product(name: "PythonKitUtilities", package: "sebbu-science")
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        )
    ]
)
