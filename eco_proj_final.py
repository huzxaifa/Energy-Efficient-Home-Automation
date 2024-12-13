import heapq
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import BOTH, LEFT, RIGHT, END, Y, X
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------- Constants ---------------------

# Energy consumption rates in watts for each device when active
DEVICE_ENERGY_CONSUMPTION = {
    'lights': 5,             # watts
    'heater': 1500,          # watts
    'air_conditioner': 1000, # watts
    'thermostat': 50,        # watts
}

# Energy pricing based on time slots (24-hour format)
ENERGY_PRICING = {
    'peak': [18, 19, 20],    # 6 PM to 8 PM
    'off_peak': [i for i in range(24) if i not in [18, 19, 20]]
}

# Total time slots (e.g., 24 for a 24-hour schedule)
TOTAL_TIME_SLOTS = 24

# External temperature profile (°C) for each time slot
# Example: Cooler at night, warmer during the day
EXTERNAL_TEMPERATURE = [
    15, 15, 15, 15, 15, 16, 18, 20, 22, 24, 25, 25,
    24, 22, 20, 18, 16, 15, 15, 15, 15, 15, 15, 15
]

# Occupancy pattern for each time slot (True if occupied, False otherwise)
# Example: Occupied from 6 AM to 9 AM and 5 PM to 10 PM
OCCUPANCY = [
    False, False, False, False, False,     # 0-4
    True, True, True,                      # 5-7 (6 AM-8 AM)
    False, False, False,                   # 8-10
    True, True, True, True, True,          # 11-15 (12 PM-4 PM)
    False, False, False,                   # 16-18
    True, True, True, True, True           # 19-23 (6 PM-11 PM)
]

# --------------------- State Class ---------------------

@dataclass(order=True)
class State:
    priority: float
    time: int
    device_states: Dict[str, Dict[str, Any]] = field(compare=False)
    temperature: float = field(compare=False)
    light_intensity: float = field(compare=False)
    cost: float = field(compare=False, default=0.0)
    parent: Optional['State'] = field(compare=False, default=None)

    def __hash__(self):
        # Create a unique identifier for the state based on time and device states
        device_state_tuple = tuple(sorted(
            (device, tuple(sorted(settings.items()))) for device, settings in self.device_states.items()
        ))
        return hash((self.time, device_state_tuple, self.temperature, self.light_intensity))

# --------------------- Helper Functions ---------------------

def get_energy_price_rate(time: int) -> float:
    """Returns the energy price for the given time slot."""
    time = time % 24  # Ensure time wraps around 24
    if time in ENERGY_PRICING['peak']:
        return 0.20  # $ per watt-hour during peak hours
    else:
        return 0.10  # $ per watt-hour during off-peak hours

def calculate_energy_cost(state: State) -> float:
    """Calculates the energy cost for the current state."""
    energy = 0.0
    for device, settings in state.device_states.items():
        if settings['status']:      #If the status of an appliance is True
            energy += DEVICE_ENERGY_CONSUMPTION[device]
    price = get_energy_price_rate(state.time)
    return energy * price / 1000  # Convert to kilowatt-hours (kWh)

def improved_heuristic(state: State, goal_time: int, preferences: Dict[str, Any]) -> float:
    """
    Improved heuristic function estimating the minimal remaining energy cost.
    Considers variable energy pricing and calculates minimal energy required for each remaining time slot.
    """
    remaining_time = goal_time - state.time
    if remaining_time <= 0:
        return 0.0
    
    heuristic_cost = 0.0
    for future_time in range(state.time, goal_time):
        price = get_energy_price_rate(future_time)
        # Minimal energy: thermostat on to maintain temperature, lights off
        min_energy = DEVICE_ENERGY_CONSUMPTION['thermostat']
        heuristic_cost += min_energy * price / 1000  # kWh
    
    return heuristic_cost

def get_possible_actions(state: State, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates all possible actions from the current state.
    Actions include adjusting thermostat settings.
    Heater and Air Conditioner are controlled automatically based on thermostat.
    """
    actions = []
    # Only allow adjusting thermostat settings
    actions.append({'device': 'thermostat', 'action': 'increase_temp'})
    actions.append({'device': 'thermostat', 'action': 'decrease_temp'})
    return actions

def apply_action(state: State, action: Dict[str, Any], preferences: Dict[str, Any], goal_time: int) -> Optional[State]:
    """
    Applies an action to the current state and returns the new state.
    Returns None if the action leads to an invalid state.
    """
    # Create a deep copy of device states to avoid mutating the original state
    new_device_states = {device: settings.copy() for device, settings in state.device_states.items()}
    
    # Extract internal temperature and thermostat setting
    internal_temp = state.temperature
    thermostat_setting = new_device_states['thermostat']['setting']
    light_intensity = state.light_intensity

    device = action['device']
    act = action['action']

    # Adjust thermostat setting based on the action
    if device == 'thermostat':
        if act == 'increase_temp':
            thermostat_setting += 1  # Increase thermostat setting by 1°C
            print(f"Thermostat increased to {thermostat_setting:.1f}°C")
        elif act == 'decrease_temp':
            thermostat_setting -= 1  # Decrease thermostat setting by 1°C
            print(f"Thermostat decreased to {thermostat_setting:.1f}°C")
        
        # Clamp thermostat setting within user preferences
        thermostat_setting = max(preferences['temperature_min'], min(preferences['temperature_max'], thermostat_setting))
        new_device_states['thermostat']['setting'] = thermostat_setting

    # Determine if heater or AC needs to be turned on/off based on internal temp vs thermostat setting
    if internal_temp < thermostat_setting:
        new_device_states['heater']['status'] = True
        new_device_states['air_conditioner']['status'] = False
        internal_temp += 0.5  # Heater effect: Increase temperature by 0.5°C
        print(f"Heater turned ON. Temperature adjusted to {internal_temp:.1f}°C")
    elif internal_temp > thermostat_setting:
        new_device_states['heater']['status'] = False
        new_device_states['air_conditioner']['status'] = True
        internal_temp -= 0.5  # AC effect: Decrease temperature by 0.5°C
        print(f"AC turned ON. Temperature adjusted to {internal_temp:.1f}°C")
    else:
        # Within desired range, turn off heater and AC
        if new_device_states['heater']['status']:
            print("Heater turned OFF.")
        if new_device_states['air_conditioner']['status']:
            print("AC turned OFF.")
        new_device_states['heater']['status'] = False
        new_device_states['air_conditioner']['status'] = False

    # Adjust internal temperature towards external temperature if heater and AC are off
    if not new_device_states['heater']['status'] and not new_device_states['air_conditioner']['status']:
        if internal_temp < EXTERNAL_TEMPERATURE[state.time]:
            internal_temp += 0.2  # Natural heating: Increase temperature by 0.2°C
            print(f"Natural heating: Temperature increased to {internal_temp:.1f}°C")
        elif internal_temp > EXTERNAL_TEMPERATURE[state.time]:
            internal_temp -= 0.2  # Natural cooling: Decrease temperature by 0.2°C
            print(f"Natural cooling: Temperature decreased to {internal_temp:.1f}°C")

    # Ensure internal temperature stays within realistic bounds
    internal_temp = max(15, min(30, internal_temp))

    # Update light intensity based on occupancy
    if OCCUPANCY[state.time]:
        new_device_states['lights']['status'] = True
        light_intensity = 300
        print(f"Lights turned ON at time slot {state.time}.")
    else:
        new_device_states['lights']['status'] = False
        light_intensity = 100
        print(f"Lights turned OFF at time slot {state.time}.")

    # Increment time slot
    new_time = state.time + 1

    # Ensure time does not exceed total slots
    if new_time > goal_time:
        return None

    # Calculate new cost
    new_cost = state.cost + calculate_energy_cost(state)

    # Create new state with updated parameters
    new_state = State(
        priority=0,  # To be set later in the A* search
        time=new_time,
        device_states=new_device_states,
        temperature=internal_temp,
        light_intensity=light_intensity,
        cost=new_cost,
        parent=state
    )

    return new_state

def is_valid_state(state: State, preferences: Dict[str, Any]) -> bool:
    """
    Checks if the state satisfies user preferences.
    Allows a small tolerance to facilitate heater and AC activation.
    """
    # Define a tolerance value (e.g., 0.5°C)
    tolerance = 0.5

    # Extract thermostat setting
    thermostat_setting = state.device_states['thermostat']['setting']

    # Check if internal temperature is within thermostat_setting ± tolerance
    if not (thermostat_setting - tolerance <= state.temperature <= thermostat_setting + tolerance):
        return False

    # Light intensity constraints
    if state.device_states['lights']['status']:
        if not (preferences['light_intensity_min'] <= state.light_intensity <= preferences['light_intensity_max']):
            return False
    else:
        # If lights are off, light intensity should be below minimum
        if state.light_intensity > preferences['light_intensity_min']:
            return False

    return True

def reconstruct_path(state: State) -> List[State]:
    """Reconstructs the path from the initial state to the given state."""
    path = []
    while state:
        path.append(state)
        state = state.parent
    return path[::-1]

# --------------------- A* Search Implementation ---------------------

def a_star_search(initial_state: State, goal_time: int, preferences: Dict[str, Any], max_states: int = 10000) -> Optional[List[State]]:
    """Performs the A* search to find the optimal energy schedule."""
    open_set = []
    heapq.heappush(open_set, initial_state)
    closed_set = set()
    states_evaluated = 0

    while open_set and states_evaluated < max_states:
        current_state = heapq.heappop(open_set)
        states_evaluated += 1

        # Debug: Print current state details
        print(f"Evaluating Time Slot: {current_state.time}, Temperature: {current_state.temperature}, Cost: {current_state.cost}")

        # Check if goal is reached
        if current_state.time >= goal_time:
            print(f"Solution found after evaluating {states_evaluated} states.")
            return reconstruct_path(current_state)

        # Create a unique identifier for the state
        state_id = hash(current_state)
        if state_id in closed_set:
            continue
        closed_set.add(state_id)

        # Generate and evaluate all possible actions
        for action in get_possible_actions(current_state, preferences):
            neighbor = apply_action(current_state, action, preferences, goal_time)
            if neighbor is None:
                continue  # Invalid state
            if not is_valid_state(neighbor, preferences):
                continue  # Skip invalid states

            # Calculate priority using cost + improved heuristic
            neighbor.cost = current_state.cost + calculate_energy_cost(current_state)
            heuristic_cost = improved_heuristic(neighbor, goal_time, preferences)
            neighbor.priority = neighbor.cost + heuristic_cost
            heapq.heappush(open_set, neighbor)

    print(f"No solution found after evaluating {states_evaluated} states.")
    return None  # No solution found

# --------------------- Tkinter UI Implementation ---------------------

def run_optimization():
    print("Starting optimization...")  # Debugging statement
    try:
        temp_min = float(entry_temp_min.get())
        temp_max = float(entry_temp_max.get())
        light_min = float(entry_light_min.get())
        light_max = float(entry_light_max.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for preferences.")
        print("Invalid input detected.")  # Debugging statement
        return

    if temp_min >= temp_max:
        messagebox.showerror("Input Error", "Minimum temperature must be less than maximum temperature.")
        print("Temperature range invalid.")  # Debugging statement
        return
    if light_min >= light_max:
        messagebox.showerror("Input Error", "Minimum light intensity must be less than maximum light intensity.")
        print("Light intensity range invalid.")  # Debugging statement
        return

    preferences = {
        'temperature_min': temp_min,
        'temperature_max': temp_max,
        'light_intensity_min': light_min,
        'light_intensity_max': light_max,
    }

    print(f"User Preferences: {preferences}")  # Debugging statement

    # Define the initial device states
    initial_device_states = {
        'lights': {'status': False},
        'heater': {'status': False},
        'air_conditioner': {'status': False},
        'thermostat': {'status': True, 'setting': (preferences['temperature_min'] + preferences['temperature_max']) / 2},
    }

    # Initialize the initial state
    initial_state = State(
        priority=0,
        time=0,
        device_states=initial_device_states,
        temperature=initial_device_states['thermostat']['setting'],
        light_intensity=100,    # Initial light intensity (lights off)
        cost=0.0,
        parent=None
    )

    goal_time = TOTAL_TIME_SLOTS
    print(f"Goal Time: {goal_time}")  # Debugging statement

    # Clear previous outputs
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Display running message
    running_label = ttk.Label(output_frame, text="Running optimization...", font=("Helvetica", 14))
    running_label.pack(pady=10)
    print("Displayed running message.")  # Debugging statement

    # Run the A* search
    optimal_path = a_star_search(initial_state, goal_time, preferences)
    print("A* search completed.")  # Debugging statement

    # Remove running message
    running_label.destroy()
    print("Removed running message.")  # Debugging statement

    if optimal_path:
        print("Optimal path found.")  # Debugging statement
        # Create Treeview for table
        table_frame = ttk.Frame(output_frame)
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        columns = ("Time Slot", "Lights Status", "Heater Status", "AC Status", "Thermostat Setting (°C)", "Temperature (°C)", "Light Intensity (lumens)", "Accumulated Cost ($)")
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')

        # Insert data into Treeview with tags
        for state in optimal_path:
            time_slot = state.time
            lights = "On" if state.device_states['lights']['status'] else "Off"
            heater = "On" if state.device_states['heater']['status'] else "Off"
            ac = "On" if state.device_states['air_conditioner']['status'] else "Off"
            thermostat_setting = f"{state.device_states['thermostat']['setting']:.1f}"
            temperature = f"{state.temperature:.1f}"
            light_intensity = f"{state.light_intensity}"
            accumulated_cost = f"{state.cost:.2f}"
            
            # Determine tag based on heater or AC status
            if heater == "On":
                tag = 'heater_on'
            elif ac == "On":
                tag = 'ac_on'
            else:
                tag = ''
            
            tree.insert("", END, values=(time_slot, lights, heater, ac, thermostat_setting, temperature, light_intensity, accumulated_cost), tags=(tag,))
        
        # Define tags for styling
        tree.tag_configure('heater_on', background='lightcoral')
        tree.tag_configure('ac_on', background='lightblue')

        tree.pack(fill=BOTH, expand=True)

        # Create Figures for graphs
        fig = Figure(figsize=(8, 4), dpi=100)

        # Temperature over Time
        ax1 = fig.add_subplot(1, 2, 1)
        times = [state.time for state in optimal_path]
        temperatures = [state.temperature for state in optimal_path]
        ax1.plot(times, temperatures, marker='o', color='red')
        ax1.set_title("Temperature Over Time")
        ax1.set_xlabel("Time Slot")
        ax1.set_ylabel("Temperature (°C)")
        ax1.grid(True)

        # Cumulative Cost over Time
        ax2 = fig.add_subplot(1, 2, 2)
        costs = [state.cost for state in optimal_path]
        ax2.plot(times, costs, marker='o', color='green')
        ax2.set_title("Cumulative Energy Cost Over Time")
        ax2.set_xlabel("Time Slot")
        ax2.set_ylabel("Cost ($)")
        ax2.grid(True)

        fig.tight_layout()

        # Embed the matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Optional: Add a total cost label
        total_cost = optimal_path[-1].cost if optimal_path else 0.0
        total_cost_label = ttk.Label(output_frame, text=f"Total Energy Cost: ${total_cost:.2f}", font=("Helvetica", 14))
        total_cost_label.pack(pady=10)
    else:
         print("No optimal schedule found.")  # Debugging statement
         messagebox.showinfo("Optimization Result", "No optimal schedule found.")

# --------------------- Main UI Setup ---------------------

def main():
    global entry_temp_min, entry_temp_max, entry_light_min, entry_light_max, output_frame

    print("Initializing main window...")  # Debugging statement

    # Create the main window
    window = tk.Tk()
    window.title("Home Automation Energy Optimization")
    window.geometry("1000x800")

    # Create a frame for user inputs
    input_frame = ttk.LabelFrame(window, text="User Preferences", padding=(20, 10))
    input_frame.pack(fill=X, padx=20, pady=10)

    # Temperature Min
    ttk.Label(input_frame, text="Minimum Temperature (°C):").grid(row=0, column=0, padx=5, pady=5, sticky='E')
    entry_temp_min = ttk.Entry(input_frame, width=10)
    entry_temp_min.grid(row=0, column=1, padx=5, pady=5)
    entry_temp_min.insert(0, "26")

    # Temperature Max
    ttk.Label(input_frame, text="Maximum Temperature (°C):").grid(row=0, column=2, padx=5, pady=5, sticky='E')
    entry_temp_max = ttk.Entry(input_frame, width=10)
    entry_temp_max.grid(row=0, column=3, padx=5, pady=5)
    entry_temp_max.insert(0, "28")

    # Light Intensity Min
    ttk.Label(input_frame, text="Minimum Light Intensity (lumens):").grid(row=1, column=0, padx=5, pady=5, sticky='E')
    entry_light_min = ttk.Entry(input_frame, width=10)
    entry_light_min.grid(row=1, column=1, padx=5, pady=5)
    entry_light_min.insert(0, "200")

    # Light Intensity Max
    ttk.Label(input_frame, text="Maximum Light Intensity (lumens):").grid(row=1, column=2, padx=5, pady=5, sticky='E')
    entry_light_max = ttk.Entry(input_frame, width=10)
    entry_light_max.grid(row=1, column=3, padx=5, pady=5)
    entry_light_max.insert(0, "500")

    # Run Button
    run_button = ttk.Button(window, text="Run Optimization", command=run_optimization)
    run_button.pack(pady=10)
    print("Run Optimization button created.")  # Debugging statement

    # Output Area Frame
    output_frame_container = ttk.Frame(window)
    output_frame_container.pack(fill=BOTH, expand=True, padx=20, pady=10)

    # Add a scrollbar to the output frame
    canvas_output = tk.Canvas(output_frame_container)
    scrollbar_output = ttk.Scrollbar(output_frame_container, orient="vertical", command=canvas_output.yview)
    output_frame = ttk.Frame(canvas_output)

    output_frame.bind(
        "<Configure>",
        lambda e: canvas_output.configure(
            scrollregion=canvas_output.bbox("all")
        )
    )

    canvas_output.create_window((0, 0), window=output_frame, anchor='nw')
    canvas_output.configure(yscrollcommand=scrollbar_output.set)

    canvas_output.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar_output.pack(side=RIGHT, fill=Y)

    # Make the window responsive
    window.columnconfigure(0, weight=1)
    window.rowconfigure(0, weight=1)

    print("Starting main loop...")  # Debugging statement
    # Start the GUI event loop
    window.mainloop()
    print("Main loop terminated.")  # Debugging statement

if __name__ == "__main__":
    main()


