import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# --- CONSTANTS ---
BOARD = 100.0
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500

# ==========================================
# GEOMETRY & PHYSICS
# ==========================================
def dist(ax, ay, bx, by): return math.hypot(ax - bx, ay - by)

def fleet_speed(ships):
    if ships <= 1: return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (max(0.0, min(1.0, ratio)) ** 1.5)

def point_to_segment_distance_sq(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0: return (px - vx)**2 + (py - vy)**2
    t = max(0.0, min(1.0, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return (px - proj_x)**2 + (py - proj_y)**2

def simulate_fleet_path(sx, sy, sr, tgt_path, tgt_r, ships, start_angle, planets, traj, target_id, src_id, max_turns=150):
    fx = sx + math.cos(start_angle) * (sr + 0.1)
    fy = sy + math.sin(start_angle) * (sr + 0.1)
    sp = fleet_speed(ships)
    vx = math.cos(start_angle) * sp
    vy = math.sin(start_angle) * sp

    for t in range(1, max_turns + 1):
        nx = fx + vx
        ny = fy + vy

        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD): return -1
        if point_to_segment_distance_sq(CENTER_X, CENTER_Y, fx, fy, nx, ny) < SUN_R**2: return -1

        for p in planets:
            if p.id == src_id: continue
            p_traj = traj[p.id]
            if t >= len(p_traj): continue

            px, py = p_traj[t-1]
            if (px - nx)**2 + (py - ny)**2 > (p.radius + 25)**2: continue

            npx, npy = p_traj[t]
            hit = False

            if point_to_segment_distance_sq(px, py, fx, fy, nx, ny) < p.radius**2: hit = True
            elif point_to_segment_distance_sq(nx, ny, px, py, npx, npy) < p.radius**2: hit = True

            if hit:
                if p.id == target_id: return t
                else: return -1

        fx, fy = nx, ny
    return -1

# ==========================================
# ORBITAL PREDICTION
# ==========================================
def predict_planet_pos(planet, initial_by_id, ang_vel, step, turns):
    init = initial_by_id.get(planet.id)
    if init is None: return planet.x, planet.y
    dx = init.x - CENTER_X
    dy = init.y - CENTER_Y
    orbital_r = math.hypot(dx, dy)
    if orbital_r + planet.radius >= 50.0: return planet.x, planet.y
    initial_angle = math.atan2(dy, dx)
    new_ang = initial_angle + ang_vel * (step + turns)
    return CENTER_X + orbital_r * math.cos(new_ang), CENTER_Y + orbital_r * math.sin(new_ang)

def get_comet_lifespan(planet_id, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids: continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        path_index = g.get("path_index", 0)
        if idx >= len(paths): return 0
        return max(0, len(paths[idx]) - path_index)
    return 500

def predict_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step, turns):
    if planet.id in comet_ids:
        for g in comets:
            pids = g.get("planet_ids", [])
            if planet.id not in pids: continue
            idx = pids.index(planet.id)
            paths = g.get("paths", [])
            path_index = g.get("path_index", 0)
            if idx >= len(paths): return None
            path = paths[idx]
            future_idx = path_index + int(turns)
            if 0 <= future_idx < len(path): return path[future_idx][0], path[future_idx][1]
            return None
        return None
    return predict_planet_pos(planet, initial_by_id, ang_vel, step, turns)

def precompute_trajectories(planets, initial_by_id, ang_vel, comets, comet_ids, step, max_turns):
    traj = {}
    for p in planets:
        path = [(p.x, p.y)]
        for t in range(1, max_turns + 1):
            pos = predict_pos(p, initial_by_id, ang_vel, comets, comet_ids, step, t)
            if pos is None: break
            path.append(pos)
        traj[p.id] = path
    return traj

# ==========================================
# THREAT MAP
# ==========================================
def build_threat_map(fleets, planets, traj, max_turns=150):
    arrivals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for f in fleets:
        sp = fleet_speed(f.ships)
        vx, vy = math.cos(f.angle) * sp, math.sin(f.angle) * sp
        fx, fy = f.x, f.y
        for t in range(1, max_turns + 1):
            nx, ny = fx + vx, fy + vy
            if not (0 <= nx <= BOARD and 0 <= ny <= BOARD): break
            if point_to_segment_distance_sq(CENTER_X, CENTER_Y, fx, fy, nx, ny) < SUN_R**2: break
            
            hit_pid = None
            for p in planets:
                p_traj = traj[p.id]
                if t >= len(p_traj): continue

                px, py = p_traj[t-1]
                if (px - nx)**2 + (py - ny)**2 > (p.radius + 25)**2: continue
                npx, npy = p_traj[t]

                if point_to_segment_distance_sq(px, py, fx, fy, nx, ny) < p.radius**2 or point_to_segment_distance_sq(nx, ny, px, py, npx, npy) < p.radius**2:
                    hit_pid = p.id
                    break
            
            if hit_pid is not None:
                arrivals[hit_pid][t][f.owner] += f.ships
                break
            else:
                fx, fy = nx, ny
    return arrivals

# ==========================================
# DISCRETE EVENT SIMULATION
# ==========================================
def simulate_planet(planet, arrivals, test_fleet=None, max_turn=100):
    owner = planet.owner
    ships = planet.ships
    prod = planet.production
    
    events = defaultdict(lambda: defaultdict(int))
    if arrivals:
        for t, arrs in arrivals.items():
            if t > max_turn: continue
            for o, s in arrs.items(): events[t][o] += s
                
    if test_fleet:
        arr_t, s, o = test_fleet
        if arr_t <= max_turn: events[arr_t][o] += s
            
    if not events:
        if owner != -1: ships += prod * max_turn
        return owner, ships
        
    actual_max = min(max_turn, max(events.keys()))
    
    for t in range(1, actual_max + 1):
        if owner != -1: ships += prod
        if t in events:
            arrs = events[t]
            att_forces = []
            for o, s in arrs.items(): att_forces.append((s, o))
                
            if att_forces:
                att_forces.sort(reverse=True)
                if len(att_forces) == 1:
                    surv_s, surv_o = att_forces[0]
                else:
                    surv_s = att_forces[0][0] - att_forces[1][0]
                    surv_o = att_forces[0][1] if surv_s > 0 else -1
                
                if surv_s > 0 and surv_o != -1:
                    if surv_o == owner: ships += surv_s
                    else:
                        if surv_s > ships: owner, ships = surv_o, surv_s - ships
                        elif surv_s == ships: owner, ships = -1, 0
                        else: ships -= surv_s
                            
    if actual_max < max_turn and owner != -1: ships += prod * (max_turn - actual_max)
    return owner, ships

def evaluate_timeline(planet, arrivals, player, remaining_steps, is_ffa, comet_lifespan, test_fleet=None):
    max_t = min(remaining_steps, comet_lifespan)
    last_event = 0
    if arrivals: last_event = max(arrivals.keys())
    if test_fleet: last_event = max(last_event, test_fleet[0])
    
    sim_t = min(max_t, last_event)
    owner, ships = simulate_planet(planet, arrivals, test_fleet, sim_t)
    
    post_turns = max_t - sim_t
    if owner != -1: ships += post_turns * planet.production
    
    if owner == player: return ships
    elif owner == -1: return 0
    else: return 0 if is_ffa else -ships

def safe_reserve(planet, arrivals, player, remaining_steps):
    low, high = 0, planet.ships
    best = planet.ships
    while low <= high:
        mid = (low + high) // 2
        dummy_p = Planet(planet.id, planet.owner, planet.x, planet.y, planet.radius, mid, planet.production)
        owner_end, _ = simulate_planet(dummy_p, arrivals, max_turn=min(150, remaining_steps))
        if owner_end == player: best, high = mid, mid - 1
        else: low = mid + 1
    return best

# ==========================================
# AIMING & VERIFICATION
# ==========================================
def precise_aim(src, tgt, ships, planets, traj, max_turns):
    tgt_path = traj[tgt.id]
    tx, ty = tgt.x, tgt.y
    for _ in range(5):
        d = max(0.0, math.hypot(tx - src.x, ty - src.y) - src.radius - 0.1 - tgt.radius)
        t = max(1, int(math.ceil(d / fleet_speed(ships))))
        if t >= len(tgt_path): return None, None
        tx, ty = tgt_path[t]

    angle = math.atan2(ty - src.y, tx - src.x)
    actual_t = simulate_fleet_path(src.x, src.y, src.radius, tgt_path, tgt.radius, ships, angle, planets, traj, tgt.id, src.id, max_turns)

    if actual_t != -1: return actual_t, angle

    expected = max(1, int(math.ceil(math.hypot(tx - src.x, ty - src.y) / fleet_speed(ships))))
    for t_offset in range(-3, 4):
        t = expected + t_offset
        if t < 1 or t >= len(tgt_path): continue
        tx, ty = tgt_path[t]
        angle = math.atan2(ty - src.y, tx - src.x)
        actual_t = simulate_fleet_path(src.x, src.y, src.radius, tgt_path, tgt.radius, ships, angle, planets, traj, tgt.id, src.id, min(max_turns, t + 2))
        if actual_t != -1: return actual_t, angle

    return None, None

def aim_and_need(src, target, arrivals, player, remaining_steps, planets, traj, initial_by_id, ang_vel, comets, comet_ids, step):
    low, high = 1, 1500
    best = None
    
    while low <= high:
        mid = (low + high) // 2
        turns, angle = precise_aim(src, target, mid, planets, traj, remaining_steps)
        if turns is None:
            high = mid - 1
            continue
            
        owner, _ = simulate_planet(target, arrivals, test_fleet=(turns, mid, player), max_turn=min(150, remaining_steps))
        if owner == player:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best is None: return None
    
    send = max(10, int(best * 1.15))
    turns, angle = precise_aim(src, target, send, planets, traj, remaining_steps)
    if turns is None: return None
    
    return send, angle, turns

# ==========================================
# MAIN AGENT
# ==========================================
def agent(obs):
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    player        = get("player", 0)
    step          = get("step", 0) or 0
    planets       = [Planet(*p) for p in get("planets", [])]
    fleets        = [Fleet(*f) for f in get("fleets", [])]
    ang_vel       = get("angular_velocity", 0.0) or 0.0
    initial_by_id = {Planet(*p).id: Planet(*p) for p in get("initial_planets", [])}
    comets        = get("comets", []) or []
    comet_ids     = set(get("comet_planet_ids", []) or [])
    my_planets    = [p for p in planets if p.owner == player]
    
    if not my_planets: return []

    remaining = max(1, TOTAL_STEPS - step)
    n_players = len(set([p.owner for p in planets if p.owner != -1]))
    is_ffa = n_players > 2
    
    traj = precompute_trajectories(planets, initial_by_id, ang_vel, comets, comet_ids, step, max_turns=250)
    arrivals = build_threat_map(fleets, planets, traj, max_turns=150)
    moves = []
    
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    frontline_status = {}
    for p in my_planets:
        min_enemy_dist = min([dist(p.x, p.y, e.x, e.y) for e in enemy_planets] + [float('inf')])
        frontline_status[p.id] = min_enemy_dist <= 50.0

    for src in my_planets:
        lifespan = get_comet_lifespan(src.id, comets) if src.id in comet_ids else 500
        res = safe_reserve(src, arrivals.get(src.id, {}), player, remaining)

        is_frontline = frontline_status[src.id]
        if is_frontline:
            res = max(res, int(src.ships * 0.15))

        available = src.ships - res
        
        # --- COMET EVACUATION PROTOCOL ---
        if lifespan <= 5:
            available = src.ships
            if available > 0:
                friends = [p for p in my_planets if p.id != src.id and p.id not in comet_ids]
                if friends:
                    best_f = min(friends, key=lambda f: dist(src.x, src.y, f.x, f.y))
                    angle = math.atan2(best_f.y - src.y, best_f.x - src.x)
                else:
                    corners = [(0,0), (0,100), (100,0), (100,100)]
                    best_c = max(corners, key=lambda c: dist(src.x, src.y, c[0], c[1]))
                    angle = math.atan2(best_c[1] - src.y, best_c[0] - src.x)
                moves.append([src.id, float(angle), int(available)])
            continue
        
        if available < 10: continue
            
        # --- SUPPLY CHAIN FUNNELING ---
        if not is_frontline:
            frontline_friends = [p for p in my_planets if p.id != src.id and p.id not in comet_ids and frontline_status.get(p.id, False)]
            if frontline_friends:
                best_f = min(frontline_friends, key=lambda f: dist(src.x, src.y, f.x, f.y))
                send = available
                turns, angle = precise_aim(src, best_f, send, planets, traj, remaining)
                if turns is not None:
                    moves.append([src.id, float(angle), int(send)])
                    arrivals[best_f.id][turns][player] += send
                    continue

        candidates = []
        for tgt in planets:
            if src.id == tgt.id: continue
            distance = dist(src.x, src.y, tgt.x, tgt.y)
            score = tgt.production / max(1.0, distance + tgt.ships * 0.25)
            if tgt.owner == -1 and distance < 40.0:
                score *= 2.0 # Boost nearby neutrals to ensure they make top 10
            elif tgt.owner != player and tgt.owner != -1:
                score *= 1.2 # Slight boost to enemies
            candidates.append((score, tgt))
            
        candidates.sort(key=lambda x: -x[0])
        
        # CPU OPTIMIZATION: Only evaluate the top 10 most valuable targets
        while available >= 10:
            best_move = None
            best_roi = -1.0
            best_tgt_obj = None
            
            for score, tgt in candidates[:10]:
                result = aim_and_need(src, tgt, arrivals.get(tgt.id, {}), player, remaining, planets, traj, initial_by_id, ang_vel, comets, comet_ids, step)
                if result is None: continue

                send, angle, turns = result
                if turns > remaining: continue
                if send > available: continue

                tgt_life = get_comet_lifespan(tgt.id, comets) if tgt.id in comet_ids else 500

                V_A = evaluate_timeline(tgt, arrivals.get(tgt.id, {}), player, remaining, is_ffa, tgt_life)
                V_B = evaluate_timeline(tgt, arrivals.get(tgt.id, {}), player, remaining, is_ffa, tgt_life, test_fleet=(turns, send, player))
                profit = (V_B - send) - V_A

                if profit > 0:
                    roi = (profit * profit) / (max(1.0, float(send)) * max(1, turns))
                    if roi > best_roi:
                        best_roi = roi
                        best_move = (tgt.id, angle, send, turns)
                        best_tgt_obj = (score, tgt)

            if best_move:
                tgt_id, angle, send, turns = best_move
                moves.append([src.id, float(angle), int(send)])
                arrivals[tgt_id][turns][player] += send
                available -= send
                candidates.remove(best_tgt_obj)
            else:
                break
                
    return moves
