// port of https://www.shadertoy.com/view/3djSzz
// the original author is FMS_Cat on Shadertoy: https://www.shadertoy.com/user/FMS_Cat
// licensed under the MIT license (the notice was not present in the original so i am not bothering either)

//const LIGHT_DIR: vec3<f32> = normalize(vec3<f32>(-3.0, 3.0, -3.0));
const INV_WAVE_LENGTH: vec3<f32> = vec3<f32>(5.60204474633241, 9.4732844379203038, 19.643802610477206);

const ESUN: f32 = 10.0;
const KR: f32 = 0.0025;
const KM: f32 = 0.0015;
const SCALE_DEPTH: f32 = 0.25;

const LIGHT_DIR: vec3<f32> = vec3<f32>(-0.57735, 0.57735, -0.57735);
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.37, 0.35, 0.34);
const SAMPLES: i32 = 2;

const G: f32 = -0.99;

const CAMERA_HEIGHT: f32 = 1.000001;

// these two can't be changed without changing `skybox__scale`

const INNER_RADIUS: f32 = 1.0;
const OUTER_RADIUS: f32 = 1.025;

fn skybox__scale(fCos: f32) -> f32{
    let x = 1.0 - fCos;
    return SCALE_DEPTH * exp( -0.00287 + x * ( 0.459 + x * ( 3.83 + x * ( -6.80 + x * 5.25 ) ) ) );
}

fn skybox__getIntersections(pos: vec3<f32>, dir: vec3<f32>, dist2: f32, rad2: f32) -> vec2<f32> {
    let B = 2.0 * dot(pos, dir);
    let C = dist2 - rad2;
    let det = max(0.0, B * B - 4.0 * C);
    return 0.5 * vec2<f32>(
        (-B - sqrt(det)),
        (-B + sqrt(det))
    );
}

fn skybox__getRayleighPhase(fCos2: f32) -> f32 {
    return 0.75 * ( 2.0 + 0.5 * fCos2 );
}

fn skybox__getMiePhase(fCos: f32, fCos2: f32, g: f32, g2: f32) -> f32 {
    return 1.5 * ( ( 1.0 - g2 ) / ( 2.0 + g2 ) ) * ( 1.0 + fCos2 )
        / pow( 1.0 + g2 - 2.0 * g * fCos, 1.5 );
}

fn skybox__uvToRayDir(uv: vec2<f32>) -> vec3<f32> {
    let v = PI * (vec2<f32>(1.5, 1.0) - vec2<f32>(2.0, 1.0) * uv);
    return vec3<f32>(
        sin(v.y) * cos(v.x),
        cos(v.y),
        sin(v.y) * sin(v.x)
    );
}

struct SkyboxConfiguration {
    light_direction: vec3<f32>,
    ground_color: vec3<f32>,
    samples: i32,

    aerosol_scattering: f32,

    camera_height: f32,
}

fn get_skybox_ray(v3RayDir: vec3<f32>) -> vec3<f32> {
    // shadertoy mock
    let iMouse = vec4<f32>(0.);
    let iResolution = vec3<f32>(480., 360., 1.);

    // Variables
    let fInnerRadius2 = INNER_RADIUS * INNER_RADIUS;
    let fOuterRadius2 = OUTER_RADIUS * OUTER_RADIUS;
    let fKrESun = KR * ESUN;
    let fKmESun = KM * ESUN;
    let fKr4PI = KR * 4.0 * PI;
    let fKm4PI = KM * 4.0 * PI;
    let fScale = 1.0 / ( OUTER_RADIUS - INNER_RADIUS );
    let fScaleOverScaleDepth = fScale / SCALE_DEPTH;
    let fG2 = G * G;

    // Light diection
    var v3LightDir = LIGHT_DIR;
    if ( 0.5 < iMouse.z ) {
		let m = iMouse.xy / iResolution.xy;
        v3LightDir = skybox__uvToRayDir( m );
    }

    let v3RayOri = vec3( 0.0, CAMERA_HEIGHT, 0.0 );
    // v3RayDir
    let fCameraHeight = length( v3RayOri );
    let fCameraHeight2 = fCameraHeight * fCameraHeight;

        let v2InnerIsects = skybox__getIntersections( v3RayOri, v3RayDir, fCameraHeight2, fInnerRadius2 );
    let v2OuterIsects = skybox__getIntersections( v3RayOri, v3RayDir, fCameraHeight2, fOuterRadius2 );
    let isGround = 0.0 < v2InnerIsects.x;

    if v2OuterIsects.x == v2OuterIsects.y { // vacuum space
        return vec3<f32>(0.);
    }

    let fNear = max( 0.0, v2OuterIsects.x );
    let fFar = select(v2OuterIsects.y, v2InnerIsects.x, isGround);
    let v3FarPos = v3RayOri + v3RayDir * fFar;
    let v3FarPosNorm = normalize( v3FarPos );

    let v3StartPos = v3RayOri + v3RayDir * fNear;
    let fStartPosHeight = length( v3StartPos );
    let v3StartPosNorm = v3StartPos / fStartPosHeight;
    let fStartAngle = dot( v3RayDir, v3StartPosNorm );
    let fStartDepth = exp( fScaleOverScaleDepth * ( INNER_RADIUS - fStartPosHeight ) );
    let fStartOffset = fStartDepth * skybox__scale( fStartAngle );

    let fCameraAngle = dot( -v3RayDir, v3FarPosNorm );
    let fCameraScale = skybox__scale( fCameraAngle );
    let fCameraOffset = exp( ( INNER_RADIUS - fCameraHeight ) / SCALE_DEPTH ) * fCameraScale;

    let fTemp = skybox__scale( dot( v3FarPosNorm, v3LightDir ) ) + skybox__scale( dot( v3FarPosNorm, -v3RayDir ) );

    let fSampleLength = ( fFar - fNear ) / f32( SAMPLES );
    let fScaledLength = fSampleLength * fScale;
    let v3SampleDir = v3RayDir * fSampleLength;
    var v3SamplePoint = v3StartPos + v3SampleDir * 0.5;

    var v3FrontColor = vec3( 0.0 );
    var v3Attenuate: vec3<f32>;
    for (var i = 0; i < SAMPLES; i++)
        {
        let fHeight = length( v3SamplePoint );
        let fDepth = exp( fScaleOverScaleDepth * ( INNER_RADIUS - fHeight ) );
        let fLightAngle = dot( v3LightDir, v3SamplePoint ) / fHeight;
        let fCameraAngle = dot( v3RayDir, v3SamplePoint ) / fHeight;

        let fScatter_if_ground = fDepth * fTemp - fCameraOffset;
        let fScatter_if_not_ground = ( fStartOffset + fDepth * ( skybox__scale( fLightAngle ) - skybox__scale( fCameraAngle ) ) );
        let fScatter = select(fScatter_if_not_ground, fScatter_if_ground, isGround);

        v3Attenuate = exp( -fScatter * ( INV_WAVE_LENGTH * fKr4PI + fKm4PI ) );
        v3FrontColor += v3Attenuate * ( fDepth * fScaledLength );
        v3SamplePoint += v3SampleDir;
    }

    v3FrontColor = clamp( v3FrontColor, vec3<f32>(0.0), vec3<f32>(3.0) );
    let c0 = v3FrontColor * ( INV_WAVE_LENGTH * fKrESun );
    let c1 = v3FrontColor * fKmESun;

    if isGround {
        let v3RayleighColor = c0 + c1;
        let v3MieColor = clamp( v3Attenuate, vec3<f32>(0.0), vec3<f32>(3.0) );
        return 1.0 - exp( -( v3RayleighColor + GROUND_COLOR * v3MieColor ) );
    }

    let fCos = dot( -v3LightDir, v3RayDir );
    let fCos2 = fCos * fCos;

    return skybox__getRayleighPhase( fCos2 ) * c0 + skybox__getMiePhase( fCos, fCos2, G, fG2 ) * c1;
}

fn get_skybox_uv(v2uv: vec2<f32>) -> vec3<f32> {
    let fRayPhi = PI * ( 3.0 / 2.0 - 2.0 * v2uv.x );
    let fRayTheta = PI * ( 1. - v2uv.y );
    let v3RayDir = vec3(
        sin( fRayTheta ) * cos( fRayPhi ),
        -cos( fRayTheta ),
        sin( fRayTheta ) * sin( fRayPhi )
    );

    return get_skybox_ray(v3RayDir);
}
